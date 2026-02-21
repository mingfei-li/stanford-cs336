from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from cs336_basics.model import scaled_dot_product_attention


@dataclass
class BenchmarkResult:
    d_model: int
    seq_len: int
    status: str
    error: str | None
    forward_mean_ms: float | None
    forward_std_ms: float | None
    backward_mean_ms: float | None
    backward_std_ms: float | None
    memory_before_backward_bytes: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch attention over a size grid")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--compile-attention", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/pytorch_attention/results.json"))
    return parser.parse_args()


def get_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_forward_timing(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_steps: int,
    measure_steps: int,
) -> tuple[float, float]:
    times_ms: list[float] = []

    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = attention_fn(q, k, v)
            maybe_sync(q.device)

        for _ in range(measure_steps):
            t0 = time.perf_counter()
            _ = attention_fn(q, k, v)
            maybe_sync(q.device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    t = torch.tensor(times_ms, dtype=torch.float64)
    return float(t.mean().item()), float(t.std(unbiased=False).item())


def run_backward_timing(
    attention_fn,
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> tuple[float, float, int | None]:
    times_ms: list[float] = []

    def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        return q, k, v

    for _ in range(warmup_steps):
        q, k, v = make_inputs()
        out = attention_fn(q, k, v)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        maybe_sync(device)

    memory_before_backward: int | None = None

    for i in range(measure_steps):
        q, k, v = make_inputs()
        out = attention_fn(q, k, v)

        if i == 0 and device.type == "cuda":
            memory_before_backward = int(torch.cuda.memory_allocated(device))

        grad_out = torch.randn_like(out)
        maybe_sync(device)
        t0 = time.perf_counter()
        out.backward(grad_out)
        maybe_sync(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    t = torch.tensor(times_ms, dtype=torch.float64)
    return float(t.mean().item()), float(t.std(unbiased=False).item()), memory_before_backward


def main() -> None:
    args = parse_args()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    attention_fn = scaled_dot_product_attention
    if args.compile_attention:
        attention_fn = torch.compile(attention_fn)

    results: list[BenchmarkResult] = []

    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)

                q = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                k = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                v = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)

                f_mean, f_std = run_forward_timing(
                    attention_fn=attention_fn,
                    q=q,
                    k=k,
                    v=v,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                )

                b_mean, b_std, mem_before_bwd = run_backward_timing(
                    attention_fn=attention_fn,
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    d_model=d_model,
                    dtype=dtype,
                    device=device,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                )

                results.append(
                    BenchmarkResult(
                        d_model=d_model,
                        seq_len=seq_len,
                        status="ok",
                        error=None,
                        forward_mean_ms=f_mean,
                        forward_std_ms=f_std,
                        backward_mean_ms=b_mean,
                        backward_std_ms=b_std,
                        memory_before_backward_bytes=mem_before_bwd,
                    )
                )
            except torch.cuda.OutOfMemoryError as e:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                results.append(
                    BenchmarkResult(
                        d_model=d_model,
                        seq_len=seq_len,
                        status="oom",
                        error=str(e),
                        forward_mean_ms=None,
                        forward_std_ms=None,
                        backward_mean_ms=None,
                        backward_std_ms=None,
                        memory_before_backward_bytes=None,
                    )
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    results.append(
                        BenchmarkResult(
                            d_model=d_model,
                            seq_len=seq_len,
                            status="oom",
                            error=str(e),
                            forward_mean_ms=None,
                            forward_std_ms=None,
                            backward_mean_ms=None,
                            backward_std_ms=None,
                            memory_before_backward_bytes=None,
                        )
                    )
                else:
                    raise

    payload = {
        "config": {
            "device": str(device),
            "batch_size": args.batch_size,
            "d_models": args.d_models,
            "seq_lens": args.seq_lens,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "dtype": args.dtype,
            "compile_attention": args.compile_attention,
        },
        "results": [asdict(r) for r in results],
    }

    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
