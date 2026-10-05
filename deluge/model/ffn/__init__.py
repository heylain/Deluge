"""FFN blocks, one per config `ffn.type`. MoE (spec 8) is M5."""

from .dense import SwiGLU

__all__ = ["SwiGLU"]
