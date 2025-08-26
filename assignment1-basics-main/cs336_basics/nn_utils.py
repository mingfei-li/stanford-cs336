import torch
from collections.abc import Callable, Iterator

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    max_logits, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - max_logits
    y = torch.gather(logits, dim=1, index=targets.unsqueeze(-1))
    z = torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
    return (z-y).mean()

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps=1e-8,
    ) -> None:
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p.grad))
                v = state.get("v", torch.zeros_like(p.grad))

                t = t + 1
                m = self.betas[0] * m + (1-self.betas[0]) * p.grad
                v = self.betas[1] * v + (1-self.betas[1]) * (p.grad**2)

                alpha_t = lr * (1 - self.betas[1]**t)**0.5 / (1 - self.betas[0]**t)
                p.data -= alpha_t * m / (v + self.eps)**0.5
                p.data -= lr * self.weight_decay * p.data
                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss
