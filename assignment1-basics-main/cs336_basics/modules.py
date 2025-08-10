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

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        initial_weights = torch.empty(
            num_embeddings,
            embedding_dim,
            device=device,
            dtype=dtype,
        )
        torch.nn.init.trunc_normal_(initial_weights, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self._weights = nn.Parameter(initial_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._weights[x]

if __name__ == "__main__":
    embedding = Embedding(10, 2)
    x = torch.LongTensor([0])
    y = embedding(x)
    print(f"Before backward: embedding._weigths.grad={embedding._weights.grad}")
    y.sum().backward()
    print(f"After backward: embedding._weigths.grad={embedding._weights.grad}")