"""Training-run configuration: budget, optimizer and checkpoint cadence.

Importable without torch, for the same reason config.py is: the budget
arithmetic (how many steps, how much accumulation, what the LR is at token N)
is needed by tooling that never builds a model.

The loader machinery is shared with the model configs rather than duplicated --
`extends:` must mean the same thing in both trees.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ..config import ConfigError, _build, _resolve

DTYPES = ("fp16", "bf16", "fp32")


@dataclass(frozen=True)
class TrainConfig:
    """One training run's budget. See configs/train/base.yaml for the meanings."""

    seq_len: int
    global_batch_tokens: int
    micro_batch: int
    total_tokens: int
    lr: float
    min_lr_ratio: float
    warmup_tokens: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    mtp_weight: float
    mtp_weight_final: float
    mtp_anneal_from: float
    mod_pred_weight: float
    router_z_weight: float
    dtype: str
    seed: int
    checkpoint_every_tokens: int
    keep_last: int
    deadline_minutes: Optional[float]

    def __post_init__(self) -> None:
        for name in ("seq_len", "micro_batch", "global_batch_tokens",
                     "total_tokens", "checkpoint_every_tokens"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")

        micro_tokens = self.micro_batch * self.seq_len
        if self.global_batch_tokens % micro_tokens:
            raise ConfigError(
                f"global_batch_tokens ({self.global_batch_tokens}) must be a whole "
                f"number of micro-batches (micro_batch * seq_len = {micro_tokens}). "
                f"A ragged final micro-batch would make the effective batch differ "
                f"from the configured one, silently, and differently on each GPU"
            )
        if self.total_tokens < self.global_batch_tokens:
            raise ConfigError(
                f"total_tokens ({self.total_tokens}) is less than one optimizer "
                f"step ({self.global_batch_tokens})"
            )
        if self.warmup_tokens >= self.total_tokens:
            raise ConfigError(
                f"warmup_tokens ({self.warmup_tokens}) must be less than "
                f"total_tokens ({self.total_tokens}); the run would never decay"
            )
        if self.warmup_tokens < 0:
            raise ConfigError("warmup_tokens must not be negative")
        if self.dtype not in DTYPES:
            raise ConfigError(
                f"unknown dtype {self.dtype!r}; known dtypes are {', '.join(DTYPES)}"
            )
        if not 0 < self.min_lr_ratio <= 1:
            raise ConfigError(
                f"min_lr_ratio ({self.min_lr_ratio}) must be in (0, 1]: it is a "
                f"fraction of lr, not an absolute learning rate"
            )
        if self.lr <= 0:
            raise ConfigError(f"lr must be positive, got {self.lr}")
        if self.grad_clip < 0:
            raise ConfigError(f"grad_clip must not be negative, got {self.grad_clip}")
        for name in ("mtp_weight", "mtp_weight_final", "mod_pred_weight",
                     "router_z_weight"):
            if getattr(self, name) < 0:
                raise ConfigError(
                    f"{name} must not be negative, got {getattr(self, name)}; "
                    f"0 disables that term")
        if not 0 <= self.mtp_anneal_from <= 1:
            raise ConfigError(
                f"mtp_anneal_from ({self.mtp_anneal_from}) is a fraction of the "
                f"run and must be in [0, 1]")
        for name in ("beta1", "beta2"):
            if not 0 <= getattr(self, name) < 1:
                raise ConfigError(f"{name} must be in [0, 1), got {getattr(self, name)}")
        if self.keep_last < 1:
            raise ConfigError(
                "keep_last must be at least 1; rotation that deletes the only "
                "checkpoint would make the run unresumable"
            )
        if self.deadline_minutes is not None and self.deadline_minutes <= 0:
            raise ConfigError(
                f"deadline_minutes ({self.deadline_minutes}) must be positive, "
                f"or null for no deadline"
            )

    # ---- derived shapes -------------------------------------------------- #

    @property
    def grad_accum(self) -> int:
        """Micro-batches per optimizer step. Derived, never configured."""
        return self.global_batch_tokens // (self.micro_batch * self.seq_len)

    @property
    def sequences_per_step(self) -> int:
        return self.global_batch_tokens // self.seq_len

    @property
    def total_steps(self) -> int:
        """Whole optimizer steps in the budget; a ragged tail step is dropped."""
        return self.total_tokens // self.global_batch_tokens

    @property
    def checkpoint_every_steps(self) -> int:
        return max(1, self.checkpoint_every_tokens // self.global_batch_tokens)

    @property
    def fingerprint(self) -> str:
        """The fields a resume may not change.

        micro_batch, keep_last, deadline_minutes and checkpoint_every_tokens are
        absent on purpose: those are how a run fits its host, and a session that
        moves from a T4 to a 4090 must be free to change them. Everything that
        defines the optimization is here, so resuming a dev run under a
        screening budget fails loudly instead of silently rescheduling the LR.
        """
        return repr((
            self.seq_len, self.global_batch_tokens, self.total_tokens,
            self.lr, self.min_lr_ratio, self.warmup_tokens, self.weight_decay,
            self.beta1, self.beta2, self.grad_clip, self.dtype, self.seed,
            self.mtp_weight, self.mtp_weight_final, self.mtp_anneal_from,
            self.mod_pred_weight, self.router_z_weight,
        ))

    # ---- schedule -------------------------------------------------------- #

    def lr_at(self, tokens_seen: int) -> float:
        """Linear warmup then cosine decay to min_lr_ratio * lr.

        A function of tokens, not of steps: resuming restores tokens_seen, so
        the schedule cannot drift if a run is restarted with a different
        micro_batch, and it stays comparable across arms whose batch shapes
        differ for VRAM reasons.
        """
        if tokens_seen < self.warmup_tokens:
            return self.lr * (tokens_seen / self.warmup_tokens)
        span = self.total_tokens - self.warmup_tokens
        progress = min(1.0, (tokens_seen - self.warmup_tokens) / span)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)

    def mtp_weight_at(self, tokens_seen: int) -> float:
        """Weight on L_mtp: flat, then annealed over the run's last stretch.

        Spec 7 says "0.3, annealed to 0.1 after 60% of training". Read as a
        linear ramp from mtp_anneal_from to the end, not a step at 60%: a step
        would move the objective discontinuously two thirds of the way through a
        run, which is exactly where a loss spike is hardest to attribute. Change
        this method, not the call sites, if the step reading turns out to be the
        intended one.

        The MTP term is worth less late because its job is to shape
        representations early; by the end the draft head is mostly trading
        against L_ce, and spec 12's A4 measures that trade directly.
        """
        start = self.mtp_anneal_from * self.total_tokens
        if tokens_seen <= start:
            return self.mtp_weight
        span = self.total_tokens - start
        if span <= 0:
            return self.mtp_weight_final
        progress = min(1.0, (tokens_seen - start) / span)
        return self.mtp_weight + (self.mtp_weight_final - self.mtp_weight) * progress


def load_train_config(path: Union[str, Path]) -> TrainConfig:
    """Build a TrainConfig from a YAML file, following `extends:`."""
    return _build(TrainConfig, _resolve(Path(path)), "train config")
