"""B3: the LFM2-style double-gated short conv.

    B, C, x~ = W_in x                    three projections
    y        = W_out(C * DWConv_k3(B * x~))

It has a learned initial conv buffer and no recurrent state, so there is
nothing for a chunk summary to read (that asymmetry is A2's, not A1's).
"""

import torch
import torch.nn as nn

from ...config import GatedConvMixer
from ..init import STD, identity_conv_, normal_
from ..state import MixerState
from .base import causal_depthwise_conv, no_gap


class GatedConv(nn.Module):
    def __init__(self, d_model: int, cfg: GatedConvMixer, out_std: float = STD):
        super().__init__()
        self.out_std = out_std
        di, k = cfg.d_inner, cfg.conv_kernel
        self.W_in = nn.Linear(d_model, 3 * di, bias=False)
        self.conv = nn.Parameter(torch.zeros(di, 1, k))
        self.W_out = nn.Linear(di, d_model, bias=False)
        self.conv_buf0 = nn.Parameter(torch.zeros(k - 1, di))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        normal_(self.W_in, STD)
        normal_(self.W_out, self.out_std)
        identity_conv_(self.conv)
        with torch.no_grad():
            self.conv_buf0.zero_()

    def forward(self, x, state=None, delta=None):
        if delta is not None:
            raise ValueError("the gated conv baseline takes no Delta (spec 10, B3)")
        if state is None:
            buffer = self.conv_buf0.expand(x.shape[0], -1, -1)
        else:
            buffer = state.conv_buf
        b_gate, c_gate, candidate = self.W_in(x).chunk(3, dim=-1)
        conv, buffer = causal_depthwise_conv(b_gate * candidate, self.conv, buffer)
        return self.W_out(c_gate * conv), MixerState(h=None, conv_buf=buffer, gap=no_gap(x))
