import math
import torch
from collections.abc import Callable, Iterable

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
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

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        return (
            min_learning_rate + 
            ((1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) / 2) *
            (max_learning_rate - min_learning_rate)
        )
    else:
        return min_learning_rate