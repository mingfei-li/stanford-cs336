from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import timeit

import torch
import torch.nn.functional as F

from cs336_basics.model import TransformerLM


MODEL_SPECS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _synchronize(device: torch.device) -> None:
    # CUDA kernels are asynchronous relative to CPU timing calls.
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _stats(samples: list[float]) -> dict[str, float | int]:
    t = torch.tensor(samples, dtype=torch.float64)
    return {
        "mean_ms": float(t.mean().item() * 1000.0),
        "std_ms": float(t.std(unbiased=False).item() * 1000.0),
        "min_ms": float(t.min().item() * 1000.0),
        "max_ms": float(t.max().item() * 1000.0),
        "num_steps": len(samples),
    }


def _next_token_loss(logits: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_tokens[:, 1:]
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_targets.reshape(-1),
        reduction="mean",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS336 A2 benchmarking script")
    parser.add_argument("--model-size", choices=list(MODEL_SPECS.keys()), default="small")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward"],
        default="forward_backward",
        help="Whether to time only forward or both forward and backward.",
    )
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="float32")
    parser.add_argument(
        "--device",
        default="cuda",
        help='Device string (e.g. "cuda", "cuda:0", "cpu"). Default: cuda.',
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable autocast mixed precision for forward/loss.",
    )
    parser.add_argument(
        "--mixed-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Autocast dtype when --mixed-precision is enabled.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Compile TransformerLM with torch.compile before benchmarking.",
    )
    parser.add_argument(
        "--include-optimizer-step",
        action="store_true",
        help="Include AdamW optimizer.step() timing in forward_backward mode.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    mixed_dtype = DTYPE_MAP[args.mixed_dtype]
    spec = MODEL_SPECS[args.model_size]

    if args.mixed_precision and device.type != "cuda":
        raise RuntimeError("Mixed precision autocast is only enabled for CUDA devices in this script.")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=spec["d_model"],
        num_layers=spec["num_layers"],
        num_heads=spec["num_heads"],
        d_ff=spec["d_ff"],
        rope_theta=args.rope_theta,
        device=device,
        dtype=dtype,
    )
    model.train()
    if args.compile_model:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(
        low=0,
        high=args.vocab_size,  # upper bound is exclusive
        size=(args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )

    def autocast_ctx():
        if args.mixed_precision:
            return torch.autocast(device_type=device.type, dtype=mixed_dtype)
        return nullcontext()

    # Warmup reduces one-time startup effects (kernel selection, allocator/cache setup).
    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            y = model(x)
            if args.mode == "forward_backward":
                loss = _next_token_loss(y, x)
        if args.mode == "forward_backward":
            loss.backward()
            if args.include_optimizer_step:
                optimizer.step()

    forward_times: list[float] = []
    backward_times: list[float] = []
    optimizer_times: list[float] = []
    total_times: list[float] = []

    for _ in range(args.measure_steps):
        optimizer.zero_grad(set_to_none=True)

        _synchronize(device)
        step_start = timeit.default_timer()

        fwd_start = timeit.default_timer()
        with autocast_ctx():
            y = model(x)
            if args.mode == "forward_backward":
                loss = _next_token_loss(y, x)
        _synchronize(device)
        fwd_end = timeit.default_timer()
        forward_times.append(fwd_end - fwd_start)

        if args.mode == "forward_backward":
            bwd_start = timeit.default_timer()
            loss.backward()
            _synchronize(device)
            bwd_end = timeit.default_timer()
            backward_times.append(bwd_end - bwd_start)

            if args.include_optimizer_step:
                opt_start = timeit.default_timer()
                optimizer.step()
                _synchronize(device)
                opt_end = timeit.default_timer()
                optimizer_times.append(opt_end - opt_start)

        _synchronize(device)
        step_end = timeit.default_timer()
        total_times.append(step_end - step_start)

    result = {
        "config": {
            "model_size": args.model_size,
            "model_spec": spec,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "vocab_size": args.vocab_size,
            "dtype": args.dtype,
            "device": str(device),
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "mode": args.mode,
            "mixed_precision": args.mixed_precision,
            "mixed_dtype": args.mixed_dtype if args.mixed_precision else None,
            "compile_model": args.compile_model,
            "include_optimizer_step": args.include_optimizer_step,
        },
        "forward": _stats(forward_times),
        "step_total": _stats(total_times),
    }
    if backward_times:
        result["backward"] = _stats(backward_times)
    if optimizer_times:
        result["optimizer_step"] = _stats(optimizer_times)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("Benchmark Results")
    print(f"  model_size: {args.model_size} ({spec})")
    print(f"  batch_size: {args.batch_size}")
    print(f"  context_length: {args.context_length}")
    print(f"  dtype/device: {args.dtype} / {device}")
    print(f"  warmup_steps: {args.warmup_steps}, measure_steps: {args.measure_steps}")
    print("")
    print(
        "  forward: "
        f"mean={result['forward']['mean_ms']:.3f} ms, "
        f"std={result['forward']['std_ms']:.3f} ms"
    )
    if "backward" in result:
        print(
            "  backward: "
            f"mean={result['backward']['mean_ms']:.3f} ms, "
            f"std={result['backward']['std_ms']:.3f} ms"
        )
    print(
        "  step_total: "
        f"mean={result['step_total']['mean_ms']:.3f} ms, "
        f"std={result['step_total']['std_ms']:.3f} ms"
    )


if __name__ == "__main__":
    main()
