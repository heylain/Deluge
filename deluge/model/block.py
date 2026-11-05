"""One block: a pre-norm residual mixer, then a pre-norm residual FFN (spec 2).

    x = x + mixer(norm1(x))
    x = x + ffn(norm2(x))

The two norms are what config.py counts as each mixer's and FFN's "pre-norm".
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .norm import RMSNorm


class Block(nn.Module):
    def __init__(self, d_model: int, mixer: nn.Module, ffn: nn.Module):
        super().__init__()
        self.norm1, self.mixer = RMSNorm(d_model), mixer
        self.norm2, self.ffn = RMSNorm(d_model), ffn

    @property
    def recurrent(self) -> bool:
        return hasattr(self.mixer, "decay")

    def forward(self, x: torch.Tensor,
                stats: Optional[Dict[str, List[torch.Tensor]]] = None) -> torch.Tensor:
        """stats, when given, collects the monitors of a recurrent mixer:
        decay_lambda [batch, heads], the mean lambda over the sequence, and
        state_rms [batch], the RMS of h at the last position."""
        normed = self.norm1(x)
        y, state = self.mixer(normed)
        if stats is not None and self.recurrent:
            # decay() again rather than a side channel out of forward(): one
            # [d_model, heads] matmul, and only on the eval path.
            stats["decay_lambda"].append(self.mixer.decay(normed).mean(dim=1))
            stats["state_rms"].append(state.h.float().pow(2).mean(dim=-1).sqrt())
        x = x + y
        return x + self.ffn(self.norm2(x))
