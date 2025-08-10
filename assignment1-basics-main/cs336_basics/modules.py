import torch
from einops import rearrange, einsum
from torch import nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._weights = nn.Parameter(
            torch.randn(
                (out_features, in_features),
                device=device,
                dtype=dtype,
            ) * (2/(in_features+out_features))**0.5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self._weights, x, "d_out d_in, ... d_in -> ... d_out")

if __name__ == "__main__":
    linear = Linear(2, 2)
    state_dict = linear.state_dict()
    print(state_dict)