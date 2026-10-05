"""B4: plain LRU -- CfC-mix without the short conv and without Delta.

    v = SiLU(W_v x), then CfC's decay, scan and gated readout, b_g included.

At A1 Delta is always 1, so CfC minus this arm is what the conv is worth
(M1 design 4.2); Delta's worth is A3's question. The parameters, their init and
decay are not copies of CfC's: both arms inherit them from cfc.GatedScanBase, so
the contrast cannot be an init difference.
"""

import torch.nn.functional as F

from ...config import LRUMixer
from ..init import STD
from ..state import MixerState
from .base import no_gap
from .cfc import GatedScanBase, gated_scan_readout


class LRU(GatedScanBase):
    def __init__(self, d_model: int, cfg: LRUMixer, out_std: float = STD):
        super().__init__(d_model, cfg, out_std)
        self.reset_parameters()

    def forward(self, x, state=None, delta=None):
        if delta is not None:
            raise ValueError("the LRU baseline takes no Delta (spec 10, B4)")
        h0 = self.h0.expand(x.shape[0], -1) if state is None else state.h
        y, h_last = gated_scan_readout(self, x, F.silu(self.W_v(x)),
                                       -self.decay(x), h0)
        return y, MixerState(h=h_last, conv_buf=None, gap=no_gap(x))
