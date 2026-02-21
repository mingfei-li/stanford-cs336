from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F

import cs336_basics.model as basics_model
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
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _next_token_loss(logits: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_tokens[:, 1:]
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_targets.reshape(-1),
        reduction="mean",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS336 A2 Nsight Systems profiling workload")
    parser.add_argument("--model-size", choices=list(MODEL_SPECS.keys()), default="small")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "train_step"],
        default="train_step",
        help="forward: forward only, forward_backward: forward+loss+backward, train_step: +optimizer step",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
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
        "--annotate-attention",
        action="store_true",
        help="Add NVTX subranges inside scaled_dot_product_attention for Nsight attribution.",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Enable CUDA memory history recording and snapshot dump.",
    )
    parser.add_argument(
        "--memory-snapshot-path",
        default="memory_snapshot.pickle",
        help="Path to write torch.cuda.memory snapshot pickle.",
    )
    parser.add_argument(
        "--memory-stats-path",
        default=None,
        help="Optional path to write peak/reserved memory stats JSON.",
    )
    return parser.parse_args()


def _install_attention_nvtx_annotations() -> None:
    def annotated_scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with nvtx.range("scaled_dot_product_attention"):
            with nvtx.range("attention_qk_matmul"):
                attention_scores = basics_model.einsum(Q, K, "... i k, ... j k -> ... i j")
            if mask is not None:
                attention_scores = torch.masked_fill(
                    attention_scores,
                    ~mask,
                    float("-inf"),
                )
            with nvtx.range("attention_softmax"):
                attention_weights = basics_model.softmax(attention_scores / (K.shape[-1] ** 0.5), -1)
            with nvtx.range("attention_pv_matmul"):
                out = basics_model.einsum(attention_weights, V, "... i j, ... j k -> ... i k")
            return out

    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def _run_single_iteration(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    mode: str,
    use_nvtx: bool,
    mixed_precision: bool,
    mixed_dtype: torch.dtype,
) -> None:
    autocast_ctx = (
        torch.autocast(device_type=x.device.type, dtype=mixed_dtype)
        if mixed_precision
        else nullcontext()
    )

    if mode != "forward":
        if use_nvtx:
            with nvtx.range("zero_grad"):
                optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)

    if use_nvtx:
        with nvtx.range("forward_pass"), autocast_ctx:
            y = model(x)
    else:
        with autocast_ctx:
            y = model(x)

    if mode in {"forward_backward", "train_step"}:
        if use_nvtx:
            with nvtx.range("loss"), autocast_ctx:
                loss = _next_token_loss(y, x)
            with nvtx.range("backward_pass"):
                loss.backward()
        else:
            with autocast_ctx:
                loss = _next_token_loss(y, x)
            loss.backward()

    if mode == "train_step":
        if use_nvtx:
            with nvtx.range("optimizer_step"):
                optimizer.step()
        else:
            optimizer.step()


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
    if args.memory_profile and device.type != "cuda":
        raise RuntimeError("Memory profiling with torch.cuda.memory requires CUDA.")

    if args.annotate_attention:
        _install_attention_nvtx_annotations()

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    x = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )

    # Warmup step(s) are intentionally not wrapped with the profile phase NVTX range.
    for _ in range(args.warmup_steps):
        _run_single_iteration(
            model,
            optimizer,
            x,
            args.mode,
            use_nvtx=False,
            mixed_precision=args.mixed_precision,
            mixed_dtype=mixed_dtype,
        )
    _synchronize(device)

    if args.memory_profile:
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    for i in range(args.profile_steps):
        with nvtx.range(f"profile_step_{i}"):
            _run_single_iteration(
                model,
                optimizer,
                x,
                args.mode,
                use_nvtx=True,
                mixed_precision=args.mixed_precision,
                mixed_dtype=mixed_dtype,
            )
            _synchronize(device)

    if args.memory_profile:
        snapshot_path = Path(args.memory_snapshot_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        torch.cuda.memory._record_memory_history(enabled=None)

        stats = {
            "mode": args.mode,
            "model_size": args.model_size,
            "context_length": args.context_length,
            "dtype": args.dtype,
            "mixed_precision": args.mixed_precision,
            "mixed_dtype": args.mixed_dtype if args.mixed_precision else None,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device=device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device=device)),
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(device=device)),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(device=device)),
        }
        if args.memory_stats_path is not None:
            stats_path = Path(args.memory_stats_path)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(json.dumps(stats, indent=2))

    print(
        f"done mode={args.mode} model={args.model_size} ctx={args.context_length} "
        f"dtype={args.dtype} mixed_precision={args.mixed_precision} "
        f"warmup={args.warmup_steps} profile_steps={args.profile_steps}"
    )


if __name__ == "__main__":
    main()
