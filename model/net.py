"""The ~215M net, exactly per docs/PRETRAIN_BRIEF.md:

12 layers x d_model 1024, 16 heads (head_dim 64), full MHA via SDPA
(is_causal=True, no custom masks), RoPE theta 10k, RMSNorm pre-norm,
QK-norm, SwiGLU hidden 2688, no biases anywhere, UNTIED embedding/head,
zero-init on residual output projections and on the head (so loss at init
is exactly ln(vocab) — the M2 init check).

Run as a script for the M2 correctness evidence:

    python model/net.py             # param count + init-loss + fwd/bwd smoke
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NetConfig:
    vocab_size: int = 32768
    seq_len: int = 2048
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    mlp_hidden: int = 2688          # 2.67x d_model rounded to /128
    rope_theta: float = 10000.0

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


def rope_cache(cfg, device):
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2,
                 dtype=torch.float32, device=device) / cfg.head_dim))
    t = torch.arange(cfg.seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv)                       # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, T, H, hd); rotate pairs (x1, x2) in the last dim
    T = x.size(1)
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos[:T, None, :], sin[:T, None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, hd = cfg.d_model, cfg.head_dim
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)         # zero-init (residual)
        self.q_norm = nn.RMSNorm(hd, eps=1e-6)
        self.k_norm = nn.RMSNorm(hd, eps=1e-6)

    def forward(self, x, cos, sin):
        B, T, d = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        q = self.q_norm(self.wq(x).view(B, T, H, hd))
        k = self.k_norm(self.wk(x).view(B, T, H, hd))
        v = self.wv(x).view(B, T, H, hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, d))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)  # zero-init

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # untied
        cos, sin = rope_cache(cfg, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for blk in self.blocks:                       # residual outputs -> 0
            nn.init.zeros_(blk.attn.wo.weight)
            nn.init.zeros_(blk.mlp.w_down.weight)
        nn.init.zeros_(self.head.weight)              # init loss == ln(vocab)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        logits = self.head(self.norm_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               targets.reshape(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_params(self):
        """Params that do a matmul per token (excludes input embedding lookup),
        for the 6N-per-token MFU estimate."""
        return self.num_params() - self.embed.weight.numel()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eot_id=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.seq_len:])
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if eot_id is not None and (nxt == eot_id).all():
                break
        return idx


if __name__ == "__main__":
    cfg = NetConfig()
    net = Net(cfg)
    n = net.num_params()
    print(f"params: {n:,} ({n/1e6:.1f}M)  [flops-active {net.flops_params()/1e6:.1f}M]")
    assert 190e6 <= n <= 230e6, "param count outside [190M, 230M]"

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    net = net.to(dev)
    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (2, 512), device=dev)
    logits, loss = net(x[:, :-1], x[:, 1:])
    want = math.log(cfg.vocab_size)
    print(f"init loss {loss.item():.4f} vs ln({cfg.vocab_size}) = {want:.4f}")
    assert abs(loss.item() - want) < 1e-2, "init loss check failed"
    loss.backward()
    gn = sum(p.grad.norm() ** 2 for p in net.parameters()
             if p.grad is not None) ** 0.5
    print(f"backward OK on {dev} (grad norm {gn:.3f})")
    print("M2 net checks pass")
