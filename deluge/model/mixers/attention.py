"""Attention for the A slots, and for every slot of A0's matched transformer.

Spec 5 without the chunk summaries, which are M2: GQA with RoPE (theta 1M) on
Q and K, a sliding window W, and n_sink learned, position-free K/V pairs per
KV head that every query sees.

The mask is explicit, [T, n_sink + T] booleans: the sinks always, plus the
window band t - W < s <= t. SDPA's is_causal cannot express it -- with the
sinks prepended its top-left alignment is off by n_sink -- and an explicit
mask gives up causal block-skipping, so every layer costs the full T^2 (the
M1 design's throughput guesses assume that).

On CUDA, SDPA is pinned to the memory-efficient backend. The math fallback
would materialise [T, n_sink + T] scores for every head and sequence; pinned,
an unsupported case fails loudly instead. The pin is the global backend switch,
set once, not the sdpa_kernel context manager: DataParallel runs its replicas
in threads, and one thread leaving the context would restore the default
backends under the other mid-call.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from ..init import STD, normal_
from ..rope import apply_rope, rope_angles

_pinned = False


def _pin_memory_efficient_backend() -> None:
    global _pinned
    if _pinned:
        return
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    _pinned = True
    print("[model] attention: SDPA pinned to the memory-efficient backend")


def attention_mask(time: int, n_sink: int, window: int,
                   device: torch.device) -> torch.Tensor:
    """[time, n_sink + time] booleans, True where query t may attend."""
    t = torch.arange(time, device=device)[:, None]
    s = torch.arange(time, device=device)[None, :]
    band = (s <= t) & (s > t - window)
    sinks = torch.ones(time, n_sink, dtype=torch.bool, device=device)
    return torch.cat([sinks, band], dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, out_std: float = STD):
        super().__init__()
        self.out_std = out_std
        self.n_heads, self.n_kv_heads = cfg.n_heads, cfg.n_kv_heads
        self.head_dim, self.window, self.n_sink = cfg.head_dim, cfg.window, cfg.n_sink
        q, kv = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
        self.W_q = nn.Linear(cfg.d_model, q, bias=False)
        self.W_k = nn.Linear(cfg.d_model, kv, bias=False)
        self.W_v = nn.Linear(cfg.d_model, kv, bias=False)
        self.W_o = nn.Linear(q, cfg.d_model, bias=False)
        self.sink_k = nn.Parameter(torch.zeros(cfg.n_kv_heads, cfg.n_sink, cfg.head_dim))
        self.sink_v = nn.Parameter(torch.zeros(cfg.n_kv_heads, cfg.n_sink, cfg.head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear in (self.W_q, self.W_k, self.W_v):
            normal_(linear, STD)
        normal_(self.W_o, self.out_std)
        with torch.no_grad():
            self.sink_k.zero_()
            self.sink_v.zero_()

    def forward(self, x, state=None, delta=None):
        if state is not None:
            raise ValueError("attention carries no state in M1: its KV cache is "
                             "the decode engine's (M4)")
        if delta is not None:
            raise ValueError("attention takes no Delta")
        batch, time, _ = x.shape
        q = self.W_q(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope_angles(torch.arange(time, device=x.device), self.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # The sinks carry no RoPE: they are position-free (spec 5).
        k = torch.cat([self.sink_k.to(k.dtype).expand(batch, -1, -1, -1), k], dim=2)
        v = torch.cat([self.sink_v.to(v.dtype).expand(batch, -1, -1, -1), v], dim=2)
        group = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        if x.is_cuda:
            _pin_memory_efficient_backend()
        mask = attention_mask(time, self.n_sink, self.window, x.device)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.W_o(y.transpose(1, 2).reshape(batch, time, -1)), None
