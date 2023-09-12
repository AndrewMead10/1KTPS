import transformer_engine as te
import torch
from torch import nn


llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


class LlamaTE(torch.nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, n_heads, num_layers, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.transformer_block = nn.Sequential(
            [
                te.pytorch.TransformerLayer(
                    hidden_size,
                    ffn_hidden_size,
                    n_heads,
                    normilization="RMSNorm",
                    activation="swiglu",
                )
                for _ in range(num_layers)
            ]
        )

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_block(x)
        x = self.lm_head(x)
        return x
