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

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = (einsum(x**2, "... d_model -> ...") / x.shape[-1] + self.eps) ** 0.5
        x = einsum(x, 1/norm,  self._weights, "... d_model, ..., d_model -> ... d_model")
        x = x.to(in_dtype)
        return x

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        d_ff = int(d_model * 8 / 3 / 64) * 64
        std = (2 / (d_model + d_ff)) ** 0.5

        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

        torch.nn.init.normal_(self.w1.data, mean=0.0, std=std)
        torch.nn.init.normal_(self.w2.data, mean=0.0, std=std)
        torch.nn.init.normal_(self.w3.data, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        x3 = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        x2 = torch.sigmoid(x1) * x1 * x3
        x = einsum(self.w2, x2, "d_model d_ff, ... d_ff -> ... d_model")
        return x

if __name__ == "__main__":
    rms_norm = RMSNorm(3)
    x = torch.ones((2, 2, 3))
    print(rms_norm(x))