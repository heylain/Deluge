"""Dense SwiGLU FFN (spec 8, v1): W_down(SiLU(W_gate x) * W_up x)."""

import torch.nn as nn
import torch.nn.functional as F

from ...config import DenseFFN
from ..init import STD, normal_


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, cfg: DenseFFN, out_std: float = STD):
        super().__init__()
        self.out_std = out_std
        self.W_gate = nn.Linear(d_model, cfg.d_ff, bias=False)
        self.W_up = nn.Linear(d_model, cfg.d_ff, bias=False)
        self.W_down = nn.Linear(cfg.d_ff, d_model, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        normal_(self.W_gate, STD)
        normal_(self.W_up, STD)
        normal_(self.W_down, self.out_std)

    def forward(self, x):
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))
