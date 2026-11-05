"""The model: embedding, n_units x (C, C, C, A) blocks, RMSNorm, tied unembedding.

build() is the factory deluge.train calls by default (the contract is in
deluge/train/trainer.py's docstring). C slots use cfg.mixer; A slots are always
attention. Imports torch only (docs/project-structure.md principle 1).
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFC_PER_UNIT, ModelConfig
from .block import Block
from .ffn import SwiGLU
from .init import STD, output_std
from .mixers import make_mixer
from .norm import RMSNorm

# Learned state rather than weights (spec 15's baked-in prompt); 2-D or 3-D,
# so the trainer's ndim rule alone would decay them toward zero.
NO_DECAY_SUFFIXES = ("conv_buf0", "sink_k", "sink_v")


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.mod is not None or cfg.mtp is not None or cfg.uses_summaries:
            raise NotImplementedError(
                "MoD, MTP and chunk summaries are M2. M1 builds the A0/A1 "
                "configs (configs/ablations/A0_*.yaml, A1_*.yaml): mod: null, "
                "mtp: null, summary_sources: []")
        if not cfg.tie_embeddings:
            raise NotImplementedError("untied embeddings: spec 2 ties them, and "
                                      "M1 builds only that")
        self.cfg = cfg
        out_std = output_std(cfg.n_blocks)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        slots = ("C",) * CFC_PER_UNIT + ("A",)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, make_mixer(cfg, slot, out_std),
                  SwiGLU(cfg.d_model, cfg.ffn, out_std))
            for _ in range(cfg.n_units) for slot in slots)
        self.norm = RMSNorm(cfg.d_model)
        nn.init.normal_(self.embed.weight, std=STD)

    def no_weight_decay(self) -> List[str]:
        """Parameter names the trainer leaves undecayed on top of its 1-D rule."""
        return [name for name, _ in self.named_parameters()
                if name.endswith(NO_DECAY_SUFFIXES)]

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                reduction: str = "mean"):
        """reduction="mean": (loss, metrics), the training contract.
        reduction="none": (token_losses [B, T], stats), the eval contract. stats
        has decay_lambda [B, n_recurrent, heads] and state_rms [B, n_recurrent],
        both absent when no layer is recurrent. A leading batch dimension on
        everything, so DataParallel's gather works."""
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        stats: Dict[str, List[torch.Tensor]] = {"decay_lambda": [], "state_rms": []}
        x = self.embed(inputs)
        for block in self.blocks:
            x = block(x, stats if reduction == "none" else None)
        logits = F.linear(self.norm(x), self.embed.weight)
        # fp32 cross-entropy: an fp16 logsumexp over 32k logits loses the small
        # differences the loss is made of.
        losses = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(),
                                 reduction="none").view_as(targets)
        if reduction == "mean":
            return losses.mean(), {}
        if not stats["decay_lambda"]:
            return losses, {}
        return losses, {"decay_lambda": torch.stack(stats["decay_lambda"], dim=1),
                        "state_rms": torch.stack(stats["state_rms"], dim=1)}


def build(cfg: ModelConfig) -> Model:
    return Model(cfg)
