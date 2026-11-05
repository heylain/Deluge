"""RMSNorm, computed in fp32.

Spec 3 puts an RMSNorm on the recurrent state h before the gated readout, and
spec 7 trains in bf16 (fp16 on Turing) with fp32 master weights. A norm whose
reduction runs in the autocast dtype loses the thing it exists to measure, so
the mean-square is accumulated in fp32 and the result cast back.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
