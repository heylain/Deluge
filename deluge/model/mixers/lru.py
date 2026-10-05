"""B4: plain LRU -- CfC-mix without the short conv and without Delta.

    v = SiLU(W_v x), then CfC's decay, scan and gated readout, b_g included.

At A1 Delta is always 1, so CfC minus this arm is what the conv is worth
(M1 design 4.2); Delta's worth is A3's question.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import LRUMixer
from ..init import GATE_BIAS, STD, decay_bias, normal_
from ..norm import RMSNorm
from ..state import MixerState
from .base import no_gap
from .cfc import gated_scan_readout


class LRU(nn.Module):
    def __init__(self, d_model: int, cfg: LRUMixer, out_std: float = STD):
        super().__init__()
        self.cfg, self.out_std = cfg, out_std
        di = cfg.d_inner
        self.n_decay_heads = cfg.n_decay_heads
        self.W_v = nn.Linear(d_model, di, bias=False)
        self.W_f = nn.Linear(d_model, cfg.n_decay_heads)
        self.W_g = nn.Linear(d_model, di)
        self.W_out = nn.Linear(di, d_model, bias=False)
        self.h0 = nn.Parameter(torch.zeros(di))
        self.norm_h = RMSNorm(di)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        normal_(self.W_v, STD)
        normal_(self.W_out, self.out_std)
        with torch.no_grad():
            self.W_f.weight.zero_()
            self.W_f.bias.copy_(decay_bias(self.cfg.a_min, self.cfg.a_max,
                                           self.cfg.n_decay_heads))
            self.W_g.weight.zero_()
            self.W_g.bias.fill_(GATE_BIAS)
            self.h0.zero_()

    def decay(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.W_f(x).float())

    def forward(self, x, state=None, delta=None):
        if delta is not None:
            raise ValueError("the LRU baseline takes no Delta (spec 10, B4)")
        h0 = self.h0.expand(x.shape[0], -1) if state is None else state.h
        y, h_last = gated_scan_readout(self, x, F.silu(self.W_v(x)),
                                       -self.decay(x), h0)
        return y, MixerState(h=h_last, conv_buf=None, gap=no_gap(x))
