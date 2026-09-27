"""Test-only model for the chain rehearsal (configs/chain/runs.yaml: smoke).

Not a language model worth training: an embedding and a linear head, just
enough to drive the trainer's real torch path -- AdamW, checkpoints, resume --
on Kaggle's CPU. The sleep makes a step's wall clock independent of whatever
CPU Kaggle hands out, which is what lets configs/train/smoke.yaml promise
exactly three sessions. build_crashing is the forced-failure arm: it raises
before the first step, which is what the chain must halt on.
"""

import time

import torch
from torch import nn
from torch.nn import functional as F

SECONDS_PER_STEP = 0.5
WIDTH = 64


class SmokeLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, WIDTH)
        self.head = nn.Linear(WIDTH, vocab_size)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        time.sleep(SECONDS_PER_STEP)
        logits = self.head(self.embed(inputs.long()))
        return F.cross_entropy(logits.flatten(0, 1), targets.long().flatten())


def build(model_cfg) -> SmokeLM:
    return SmokeLM(model_cfg.vocab_size)


def build_crashing(model_cfg):
    raise RuntimeError("smoke-crash: deliberate failure, for the chain's halt path")
