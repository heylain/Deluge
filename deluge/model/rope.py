"""Rotary position embedding, theta = 1M (spec 5).

Channel pairs are adjacent, (2i, 2i + 1), as in the RoPE paper, and pair i
turns by angle t * theta^(-2i / head_dim) at position t. The angles are formed
in float64 and the rotation is done in fp32, then cast back: at t = 4095 an
fp16 angle would be off by whole radians. tests/test_attention.py checks this
against a complex-number form that shares no code with it.
"""

import torch

THETA = 1_000_000.0


def rope_angles(positions: torch.Tensor, head_dim: int,
                theta: float = THETA) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin), each [len(positions), head_dim // 2], fp32."""
    if head_dim % 2:
        raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float64,
                             device=positions.device) / head_dim
    angles = positions.to(torch.float64)[:, None] * theta ** -exponents[None, :]
    return torch.cos(angles).float(), torch.sin(angles).float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x [..., time, head_dim] by the angles for its positions."""
    pairs = x.float().unflatten(-1, (-1, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)
