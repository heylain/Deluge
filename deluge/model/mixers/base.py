"""The mixer interface: docs/project-structure.md calls this file "the whole
ablation strategy".

Every slot mixer -- CfC, the gated conv, the LRU, attention -- is

    forward(x, state=None, delta=None) -> (y, state)

with x the pre-normed residual stream [batch, time, d_model].

  state=None  a fresh sequence: a recurrent mixer starts from its learned h0
              and conv buffer, broadcast over the batch. The returned state
              (a MixerState, deluge/model/state.py) continues the sequence
              exactly: tests/test_mixers.py splits a sequence in two to check.
  delta=None  Delta = 1, the only value M1 uses. MoD's gaps are M2.

Recurrent mixers also have decay(x) -> lambda [batch, time, heads], which the
model reads for spec 7's stability monitors.
"""

import torch
import torch.nn.functional as F


def causal_depthwise_conv(u: torch.Tensor, weight: torch.Tensor,
                          buffer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A depthwise causal conv that carries its last k - 1 inputs.

    u [batch, time, channels]; weight [channels, 1, k]; buffer
    [batch, k - 1, channels], the inputs before u. Returns (the conv output
    [batch, time, channels], the new buffer).
    """
    k = weight.shape[-1]
    padded = torch.cat([buffer.to(u.dtype), u], dim=1)
    out = F.conv1d(padded.transpose(1, 2), weight, groups=u.shape[-1])
    return out.transpose(1, 2), padded[:, padded.shape[1] - (k - 1):]


def no_gap(x: torch.Tensor) -> torch.Tensor:
    """MixerState.gap for a sequence MoD never skips: zero tokens skipped."""
    return torch.zeros(x.shape[0], device=x.device)
