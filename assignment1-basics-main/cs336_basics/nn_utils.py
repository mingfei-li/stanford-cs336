import torch
from collections.abc import Iterable

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    max_logits, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - max_logits
    y = torch.gather(logits, dim=1, index=targets.unsqueeze(-1))
    z = torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
    return (z-y).mean()

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    parameters = list(parameters)
    grad_norm_squared = 0.
    for param in parameters:
        if param.grad is not None:
            grad_norm_squared += (param.grad ** 2).sum()
    multiplier = min(max_l2_norm / (grad_norm_squared ** 0.5 + 1e-6), 1.0)
    for param in parameters:
        if param.grad is not None:
            param.grad *= multiplier
