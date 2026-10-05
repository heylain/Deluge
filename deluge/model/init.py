"""Initialisation (spec 3, M1 design 4.4).

Every module calls these from its own reset_parameters, so a mixer built on
its own is initialised exactly as it is inside the model.

  b_f     softplus^-1 of a geometric grid of lambda over heads, from -ln a_max
          to -ln a_min (design D6): log-uniform in the time constant. Dev's 12
          heads remember 999, 655, 429, 281, 184, 120, 79, 52, 34, 22, 14 and
          9 tokens.
  W_f     zero, so the designed decays are exact at step 0. W_f still gets
          gradient through x.
  W_g     zero, with b_g = GATE_BIAS: the gate is SiLU(1.278) = 0.9995 at step
          0, with gradient 1.000 (design D7).
  convs   identity on the current-token tap, the spec's "centre tap" read
          causally. CfC's k4 and the gated conv's k3 alike, so A1 does not also
          compare init schemes.
  zero    h0, the initial conv buffers, and the attention sinks' K and V. A
          zero sink has logit 0 and contributes nothing, so attention starts as
          softmax with n_sink extra units in the denominator (the spec leaves
          sink init open; this is the M1 choice).
  normal  std STD for linears and the embedding; output projections (W_out,
          W_o, the FFN's W_down) std STD / sqrt(2 * n_blocks).
  ones    RMSNorm gains (RMSNorm's own init).
"""

import math

import torch
from torch import nn

STD = 0.02
GATE_BIAS = 1.278


def output_std(n_blocks: int) -> float:
    """Residual-branch output projections, scaled so the stream's variance does
    not grow with depth."""
    return STD / math.sqrt(2 * n_blocks)


def decay_rates(a_min: float, a_max: float, n_heads: int) -> torch.Tensor:
    """lambda per head, float64: a geometric grid from -ln a_max (the longest
    memory, head 0) to -ln a_min."""
    slow, fast = -math.log(a_max), -math.log(a_min)
    if n_heads == 1:
        return torch.tensor([slow], dtype=torch.float64)
    steps = torch.arange(n_heads, dtype=torch.float64) / (n_heads - 1)
    return slow * (fast / slow) ** steps


def decay_bias(a_min: float, a_max: float, n_heads: int) -> torch.Tensor:
    """b_f = softplus^-1(lambda) = log(expm1(lambda)), so softplus(b_f) is the
    grid. expm1, because lambda is as small as 0.001."""
    return torch.log(torch.expm1(decay_rates(a_min, a_max, n_heads))).float()


def normal_(linear: nn.Linear, std: float) -> None:
    nn.init.normal_(linear.weight, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def identity_conv_(weight: torch.Tensor) -> None:
    """Depthwise conv weight [channels, 1, k]: 1 on the last tap, which
    F.conv1d applies to the current token, and 0 on the k - 1 before it."""
    with torch.no_grad():
        weight.zero_()
        weight[..., -1] = 1.0
