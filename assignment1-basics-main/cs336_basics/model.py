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
        self.weight = nn.Parameter(
            torch.randn(
                (out_features, in_features),
                device=device,
                dtype=dtype,
            ) * (2/(in_features+out_features))**0.5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

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
        self.weight = nn.Parameter(initial_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = (einsum(x**2, "... d_model -> ...") / x.shape[-1] + self.eps) ** 0.5
        x = einsum(x, 1/norm,  self.weight, "... d_model, ..., d_model -> ... d_model")
        x = x.to(in_dtype)
        return x

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 8 / 3 / 64) * 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        x2 = torch.sigmoid(x1) * x1 * x3
        x = self.w2(x2)
        return x

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        theta_ik = einsum(
            torch.arange(0, max_seq_len), 
            theta ** (-torch.arange(0, d_k//2)*2 / d_k),
            "i, k -> i k"
        ).to(device)
        cos_ik = torch.cos(theta_ik)
        sin_ik = torch.sin(theta_ik)
        rotation_matrices = rearrange(
            [cos_ik, -sin_ik, sin_ik, cos_ik],
            "(h w) i k -> i k h w",
            h=2,
        )
        self.register_buffer(
            "rotation_matrices",
            rotation_matrices,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        r = self.rotation_matrices[token_positions]

        x = rearrange(x, "... t (g i) -> ... t g i", i=2)
        x = einsum(r, x, "... t g j i, ... t g i -> ... t g j")
        x = rearrange(x, "... t g j -> ... t (g j)")

        return x

def softmax(x: torch.Tensor, dim: int) -> None:
    max_value, _ = x.max(dim=dim, keepdim=True)
    x = x - max_value
    x = x.exp()
    x = x / x.sum(dim=dim, keepdim=True)
    return x

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attention_scores = einsum(Q, K, "... i k, ... j k -> ... i j")
    if mask is not None:
        attention_scores = torch.masked_fill(
            attention_scores,
            ~mask,
            float("-inf"),
        )
    attention_weights = softmax(attention_scores / (K.shape[-1] ** 0.5), -1)
    out = einsum(attention_weights, V, "... i j, ... j k -> ... i k")
    return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        rope_embedding: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = num_heads
        self.rope_emb = rope_embedding
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        self.device = device

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor: 
        seq_len = x.shape[-2]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(
            q,
            "... seq_len (n_heads d_head) -> ... n_heads seq_len d_head",
            n_heads=self.n_heads,
        )
        k = rearrange(
            k,
            "... seq_len (n_heads d_head) -> ... n_heads seq_len d_head",
            n_heads=self.n_heads,
        )
        v = rearrange(
            v,
            "... seq_len (n_heads d_head) -> ... n_heads seq_len d_head",
            n_heads=self.n_heads,
        )
        if self.rope_emb is not None:
            if token_positions is None:
                token_positions = torch.arange(0, seq_len)
            q = self.rope_emb(q, token_positions)
            k = self.rope_emb(k, token_positions)
        mask = torch.tril(torch.ones((seq_len, seq_len))).bool().to(self.device)
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(
            out,
            "... n_heads seq_len d_head -> ... seq_len (n_heads d_head)"
        )
        out = self.output_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        d_head = d_model // num_heads
        rope_emb = RotaryPositionalEmbedding(theta, d_head, max_seq_len, device)
        self.ln1 = RMSNorm(d_model, device, dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope_emb, device, dtype)
        self.ln2 = RMSNorm(d_model, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x + self.attn(self.ln1(x))
        # x = x + self.ffn(self.ln2(x))
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                rope_theta,
                device,
                dtype,
            ) for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device, dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        # x = self.ln_final(x)
        x = self.lm_head(x)
        return x

if __name__ == "__main__":
    rope = RotaryPositionalEmbedding(4/torch.pi, 2, 5)
    print(rope.rotation_matrices)