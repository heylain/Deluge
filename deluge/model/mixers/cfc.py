"""CfC-mix (spec 3), the layer under test.

    v     = SiLU(DWConv_k4(W_v x))
    lam   = softplus(W_f x + b_f)              per head, broadcast over its channels
    log a = -Delta * lam,   b = sqrt(1 - a^2)
    h     = scan(log a, b * v) from h0
    y     = W_out(RMSNorm(h) * SiLU(W_g x + b_g))

lambda, b and the scan run in fp32 (spec 3's numerics); the projections run
in whatever autocast says. b_g is M1 design D7: on zero-mean pre-normed input
SiLU(W_g x) averages 0.07 whatever W_g is, so only a bias can start the gate
near 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import CfCMixer
from ...kernels.head_scan import head_chunked_scan
from ...kernels.reference.scan import input_scale
from ..init import GATE_BIAS, STD, decay_bias, identity_conv_, normal_
from ..norm import RMSNorm
from ..state import MixerState
from .base import causal_depthwise_conv, no_gap

# The scan's tile, not architecture: it is not cfg.chunk (the summary chunk),
# and changing it changes only float rounding, memory and speed (design 5).
SCAN_CHUNK = 64


def gated_scan_readout(mixer: nn.Module, x: torch.Tensor, v: torch.Tensor,
                       log_a: torch.Tensor, h0: torch.Tensor):
    """The part CfC and the LRU share: scan b * v from h0, then the gated
    readout. Returns (y, h after the last token)."""
    b = input_scale(log_a)                                        # [B, T, H] fp32
    per_head = v.shape[-1] // log_a.shape[-1]
    bv = b.repeat_interleave(per_head, dim=-1) * v.float()
    states, h_last = head_chunked_scan(log_a, bv, h0, SCAN_CHUNK)
    y = mixer.W_out(mixer.norm_h(states) * F.silu(mixer.W_g(x)))
    return y, h_last


class CfC(nn.Module):
    def __init__(self, d_model: int, cfg: CfCMixer, out_std: float = STD):
        super().__init__()
        self.cfg, self.out_std = cfg, out_std
        di, k = cfg.d_inner, cfg.conv_kernel
        self.n_decay_heads = cfg.n_decay_heads
        self.W_v = nn.Linear(d_model, di, bias=False)
        self.conv = nn.Parameter(torch.zeros(di, 1, k))          # depthwise
        self.W_f = nn.Linear(d_model, cfg.n_decay_heads)          # bias is b_f
        self.W_g = nn.Linear(d_model, di)                         # bias is b_g
        self.W_out = nn.Linear(di, d_model, bias=False)
        self.h0 = nn.Parameter(torch.zeros(di))
        self.conv_buf0 = nn.Parameter(torch.zeros(k - 1, di))
        self.norm_h = RMSNorm(di)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        normal_(self.W_v, STD)
        normal_(self.W_out, self.out_std)
        identity_conv_(self.conv)
        with torch.no_grad():
            self.W_f.weight.zero_()
            self.W_f.bias.copy_(decay_bias(self.cfg.a_min, self.cfg.a_max,
                                           self.cfg.n_decay_heads))
            self.W_g.weight.zero_()
            self.W_g.bias.fill_(GATE_BIAS)
            self.h0.zero_()
            self.conv_buf0.zero_()

    def decay(self, x: torch.Tensor) -> torch.Tensor:
        """lambda [batch, time, heads], fp32."""
        return F.softplus(self.W_f(x).float())

    def forward(self, x, state=None, delta=None):
        batch = x.shape[0]
        if state is None:
            h0 = self.h0.expand(batch, -1)
            buffer = self.conv_buf0.expand(batch, -1, -1)
        else:
            h0, buffer = state.h, state.conv_buf
        conv, buffer = causal_depthwise_conv(self.W_v(x), self.conv, buffer)
        lam = self.decay(x)
        log_a = -lam if delta is None else -delta.unsqueeze(-1).float() * lam
        y, h_last = gated_scan_readout(self, x, F.silu(conv), log_a, h0)
        return y, MixerState(h=h_last, conv_buf=buffer, gap=no_gap(x))
