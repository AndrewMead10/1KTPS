from dataclasses import dataclass
from torch import nn
import torch
from triton_functions import rbe_triton_wrapper, rmsnorm_triton_wrapper
import torch.nn.functional as F
from flash_attn import flash_attn_func


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 32000
    norm_eps: float = 1e-6
    max_seq_length: int = 2048


DEBUG_CONFIG = ModelArgs(
    dim=32,
    n_layers=10,
    n_heads=4,
    vocab_size=32000,
)
LLAMA_7B_CONFIG = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    vocab_size=32000,
)
LLAMA_13B_CONFIG = ModelArgs(
    dim=5120,
    n_layers=40,
    n_heads=40,
    vocab_size=32000,
)

LLAMA_CONFIG_DICT = {
    "7B": LLAMA_7B_CONFIG,
    "13B": LLAMA_13B_CONFIG,
    "debug": DEBUG_CONFIG,
}

MULTIPLE_OF = 256

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_triton_wrapper(x, self.weight, self.eps)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class Attention(nn.Module):
    def __init__(self, n_heads: int, dim: int):
        super().__init__()

        self.n_local_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(B, T, self.n_local_heads, self.head_dim)
        xk = xk.view(B, T, self.n_local_heads, self.head_dim)
        xv = xv.view(B, T, self.n_local_heads, self.head_dim)

        xq = rbe_triton_wrapper(xq)
        xk = rbe_triton_wrapper(xk)

        if self.training:
            dropout = 0.1
        else:
            dropout = 0.0

        attn_out = flash_attn_func(xq, xk, xv, dropout=dropout, causal=True)

        attn_out = attn_out.view(B, T, C)

        return self.wo(attn_out)
    
class TransformerBlock(nn.Module):
    def __init__(self, n_heads: int, dim: int, norm_eps: float):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(n_heads, dim)
        self.feed_forward = FeedForward(
            dim=dim, hidden_dim=4 * dim, multiple_of=MULTIPLE_OF
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x))
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(params.n_heads, params.dim, params.norm_eps))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

    def forward(self, tokens: torch.Tensor):
        _bsz, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        output = self.output(h)
        return output.float()


model = Transformer(DEBUG_CONFIG)



