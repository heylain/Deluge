"""Per-sequence recurrent state.

One definition, four consumers: the training scan needs it at chunk boundaries,
the decode loop carries it token to token, snapshots persist it, and speculative
decoding clones and restores it. If any two of those disagree about what state
is, the golden tests of spec 12 fail -- which is why this is a type and not a
tuple passed around by convention.

Tensor-only and pickle-safe, so a snapshot can go to disk (spec 9) without a
custom serialiser.
"""

from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass
class MixerState:
    """State one mixer block carries between tokens.

    h            recurrent state, [batch, d_inner]. None for a stateless mixer
                 (B3's gated conv), which is why chunk summaries fall back to
                 the residual stream in that arm.
    conv_buf     the last k-1 inputs to the depthwise conv, [batch, k-1, d_inner].
                 None for a mixer without a conv (B4's LRU).
    gap          tokens skipped since this block last processed one, [batch].
                 Spec 6: the next processed token uses Delta = 1 + gap. Always
                 present, and zero when MoD is off, so the Delta path is one
                 code path rather than two.
    """

    h: Optional[torch.Tensor]
    conv_buf: Optional[torch.Tensor]
    gap: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.gap.shape[0]

    def snapshot(self) -> "MixerState":
        """A detached deep copy.

        Speculative decoding (spec 9) takes one of these at every draft position
        during verification and restores the last accepted one. The recurrence
        cannot be run backwards, so this copy is the only way back.
        """
        return MixerState(
            h=None if self.h is None else self.h.detach().clone(),
            conv_buf=None if self.conv_buf is None else self.conv_buf.detach().clone(),
            gap=self.gap.detach().clone(),
        )

    def index_select(self, index: torch.Tensor) -> "MixerState":
        """Gather a subset of the batch, for rollback and for beam reordering."""
        return MixerState(
            h=None if self.h is None else self.h.index_select(0, index),
            conv_buf=(None if self.conv_buf is None
                      else self.conv_buf.index_select(0, index)),
            gap=self.gap.index_select(0, index),
        )

    def to(self, *args, **kwargs) -> "MixerState":
        moved = {name: (None if value is None else value.to(*args, **kwargs))
                 for name, value in vars(self).items()}
        return replace(self, **moved)

    def detach(self) -> "MixerState":
        """Cut the graph at a chunk boundary without copying."""
        return MixerState(
            h=None if self.h is None else self.h.detach(),
            conv_buf=None if self.conv_buf is None else self.conv_buf.detach(),
            gap=self.gap.detach(),
        )
