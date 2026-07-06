"""GPT: token+position embeddings -> N causal self-attention blocks -> LM head.

A decoder-only Transformer (GPT-2 style). Run as a script for a shape/loss
smoke test:

    .venv/bin/python3 src/model.py
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    block_size: int = 256      # max sequence length (must match the data-loader window)
    n_embd: int = 1024         # width of every vector in the model
    n_head: int = 16           # attention heads (n_embd must divide by this)
    n_layer: int = 14          # number of stacked blocks -> ~210M params
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Each token attends to itself and earlier tokens (never the future)."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # one projection makes query, key, value for ALL heads at once (hence 3x).
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # projection applied after mixing the heads back together.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask: a lower-triangular matrix of 1s. Position t may look at
        # columns 0..t only. Stored as a buffer (not a learnable parameter).
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.shape                       # batch, seq len, n_embd
        head_dim = C // self.n_head

        # 1) project to q, k, v and split them apart
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2) reshape each into separate heads: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # 3) scores = query . key, scaled so softmax stays well-behaved
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)   # (B, nh, T, T)

        # 4) causal mask: blank out the future BEFORE softmax
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # 5) turn scores into weights that sum to 1, then average the values
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                             # (B, nh, T, head_dim)

        # 6) reassemble the heads back into one vector per token
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Per-token feed-forward: widen 4x, non-linearity, project back."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One Transformer block: attention then MLP, each with a residual add.

    Pre-norm: LayerNorm is applied to the INPUT of each sub-layer, and the
    sub-layer's output is ADDED back to x (the residual). Residuals let
    gradients flow cleanly through a deep stack.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))    # attend, then add back
        x = x + self.mlp(self.ln_2(x))     # think per-token, then add back
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token table
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # position table
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)                     # final norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: the input embedding and output projection share one
        # matrix. Saves ~33M params and usually helps quality.
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"sequence length {T} > block_size {self.config.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))   # (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # flatten (B, T) into one long list of predictions vs. targets
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss


if __name__ == "__main__":
    from data import get_batch

    config = GPTConfig()
    model = GPT(config)
    x, y = get_batch("train", batch_size=4, block_size=config.block_size)

    logits, loss = model(x, y)
    n_params = sum(p.numel() for p in model.parameters())

    print("input ids shape:", tuple(x.shape))          # (4, 256)
    print("logits shape:   ", tuple(logits.shape))     # (4, 256, 32768)
    print(f"param count:     {n_params:,}")
    print(f"initial loss:    {loss.item():.3f}")
    print(f"expected (~random): ln(vocab) = {math.log(config.vocab_size):.3f}")
