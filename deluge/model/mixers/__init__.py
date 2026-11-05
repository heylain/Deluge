"""Slot mixers, one per config `mixer.type` (deluge/config.py MIXER_TYPES)."""

from torch import nn

from ...config import (AttentionMixer, CfCMixer, GatedConvMixer, LRUMixer,
                       ModelConfig)
from .attention import Attention
from .cfc import CfC
from .gated_conv import GatedConv
from .lru import LRU

__all__ = ["Attention", "CfC", "GatedConv", "LRU", "make_mixer"]


def make_mixer(cfg: ModelConfig, slot: str, out_std: float) -> nn.Module:
    """The mixer for a "C" slot (cfg.mixer) or an "A" slot (always attention)."""
    mixer = cfg.mixer if slot == "C" else AttentionMixer()
    if isinstance(mixer, CfCMixer):
        return CfC(cfg.d_model, mixer, out_std)
    if isinstance(mixer, LRUMixer):
        return LRU(cfg.d_model, mixer, out_std)
    if isinstance(mixer, GatedConvMixer):
        return GatedConv(cfg.d_model, mixer, out_std)
    if isinstance(mixer, AttentionMixer):
        return Attention(cfg, out_std)
    raise TypeError(f"no module for mixer config {mixer!r}")
