import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    max_logits, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - max_logits
    y = torch.gather(logits, dim=1, index=targets.unsqueeze(-1))
    z = torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
    return (z-y).mean()
