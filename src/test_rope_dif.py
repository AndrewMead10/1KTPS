import torch
from torch import nn
from typing import Optional, Tuple
from einops import rearrange


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cpu",
    base: int = 10000,
) -> torch.Tensor:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (
        base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem)
    )

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    x_pass = x[:, :, :, 32:]
    x_rot = x[:, :, :, :32]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rope_cache = rope_cache.unsqueeze(1)

    cos = rope_cache[..., 0]
    sin = rope_cache[..., 1]

    x_out2 = torch.cat(
        [
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ],
        -1,
    )

    return torch.cat((x_out2, x_pass), dim=-1).type_as(x)


class RotaryEmbedding(nn.Module):
    """PyTorch implementation of `flash-attn` RotaryEmbedding layer.
    Adapted from https://github.com/Dao-AILab/flash-attention."""

    def __init__(
        self,
        dim: int,
        base: Optional[int] = 10000,
        scale_base: Optional[float] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Generate and save the inverse frequency buffer (non-trainable)
        self.dim = dim
        self.base = base
        self.scale_base = scale_base
        self.device = device

        self.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(
        self, x: torch.FloatTensor, seqlen_offset: Optional[int] = 0
    ) -> None:
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        seqlen = x.shape[1] + seqlen_offset

        # Re-generate the inverse frequency buffer if it's not fp32
        # (for instance if model.half() was called)

        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=torch.float32)

            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(
                t, self.inv_freq.to(device=t.device, dtype=torch.float32)
            )
            self._cos_cached = torch.cos(freqs).to(x.dtype)
            self._sin_cached = torch.sin(freqs).to(x.dtype)

    def apply_rotary_emb_qkv(
        self,
        qkv: torch.FloatTensor,
        sin: torch.FloatTensor,
        cos: torch.FloatTensor,
        sin_k: Optional[torch.FloatTensor] = None,
        cos_k: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        _, seqlen, three, _, headdim = qkv.shape
        assert three == 3

        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen

        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert (
            sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        )

        q_rot = qkv[:, :, 0, :, :rotary_dim]
        q_pass = qkv[:, :, 0, :, rotary_dim:]

        k_rot = qkv[:, :, 1, :, :rotary_dim]
        k_pass = qkv[:, :, 1, :, rotary_dim:]

        # Splits the queries and keys in half
        q1, q2 = q_rot.chunk(2, dim=-1)
        k1, k2 = k_rot.chunk(2, dim=-1)
        c, s = rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(
            sin[:seqlen], "s d -> s 1 d"
        )

        # Casts to fp32 are necessary to prevent fp16 overflow issues
        q1, q2, k1, k2, c, s = [
            t.to(dtype=torch.float32) for t in [q1, q2, k1, k2, c, s]
        ]

        # Computes the new keys and queries, recasting to original dtype
        q_rot = torch.cat([q1 * c - q2 * s, q1 * s + q2 * c], axis=-1).to(qkv.dtype)

        # k_rot = torch.cat([k1 * c - k2 * s, k1 * s + k2 * c], axis=-1).to(qkv.dtype)

        return torch.cat([q_rot, q_pass], axis=-1)

    def forward(
        self, qkv: torch.Tensor, seqlen_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_cache(qkv, seqlen_offset)

        return self.apply_rotary_emb_qkv(
            qkv, self._sin_cached[seqlen_offset:], self._cos_cached[seqlen_offset:]
        )


chili_rope = build_rope_cache(1024, 32)

# print(chili_rope[:, :, 0].shape)
# print(chili_rope[-1])
x_phi = torch.randn((1, 1024, 3, 32, 64))


phi_rope = RotaryEmbedding(32)

# print(x_phi[:, :, 0].shape)
x1 = apply_rope(x_phi[:, :, 0], chili_rope)
# print(x1.shape)
x2 = phi_rope(x_phi)

# print(torch.allclose(x1[0], x2[0]))
# print(torch.allclose(x1[1], x2[1]))
# print(x1[2].shape, x2[2].shape)
# print(torch.allclose(x1[2], x2[2]))
# print(torch.allclose(x1[3], x2[3]))


print(torch.allclose(x1, x2))

# print(x1)
# print(x2)


# # x2 = x2[:, :, 0]
# print(torch.allclose(x1, x2))

# print(phi_rope._cos_cached.shape)
# print(torch.allclose(phi_rope._cos_cached, chili_rope[:, :, 0]))
# # print(phi_rope._sin_cached)
