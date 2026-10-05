"""Held-out eval accounting and the stability monitors (spec 7's watch-list,
M1 design 7.1-7.2).

Torch-free, like loop.py. The trainer's evaluator feeds EvalAccumulator numpy
arrays, so the arithmetic the "stable" criteria and the A1 verdict read --
position buckets, the decay summary, the skip count -- is testable without a
model.
"""

from typing import Any, Dict, Optional

import numpy as np

# The in-run evals read the first 488 pieces of heldout.bin: 488 x 4096 is
# ~2M tokens, and heldout.bin's hash order makes them a random document sample.
SUBSET_PIECES = 488
# Position ranges the A1 writeup reads: a CfC advantage from long-range state
# should show in the last bucket, beyond the 512-token windows' reach.
BUCKETS = ((0, 512), (512, 1536), (1536, 4096))


class EvalAccumulator:
    """Sums over held-out batches; result() turns them into an eval record."""

    def __init__(self) -> None:
        self.loss_sum, self.tokens, self.sequences = 0.0, 0, 0
        self.bucket_sum = {bucket: 0.0 for bucket in BUCKETS}
        self.bucket_tokens = {bucket: 0 for bucket in BUCKETS}
        self.lambda_sum: Optional[np.ndarray] = None    # [layers, heads]
        self.rms_sum: Optional[np.ndarray] = None       # [layers]

    def add(self, losses: np.ndarray, stats: Dict[str, np.ndarray]) -> None:
        """losses [batch, time] in nats; stats as the model returns them
        (decay_lambda [batch, layers, heads], state_rms [batch, layers]), or
        empty for an arm with no recurrent layer."""
        losses = np.asarray(losses, dtype=np.float64)
        self.loss_sum += losses.sum()
        self.tokens += losses.size
        self.sequences += losses.shape[0]
        for lo, hi in BUCKETS:
            part = losses[:, lo:hi]
            self.bucket_sum[(lo, hi)] += part.sum()
            self.bucket_tokens[(lo, hi)] += part.size
        if "decay_lambda" in stats:
            lam = np.asarray(stats["decay_lambda"], dtype=np.float64).sum(axis=0)
            rms = np.asarray(stats["state_rms"], dtype=np.float64).sum(axis=0)
            self.lambda_sum = lam if self.lambda_sum is None else self.lambda_sum + lam
            self.rms_sum = rms if self.rms_sum is None else self.rms_sum + rms

    def result(self) -> Dict[str, Any]:
        if not self.tokens:
            raise ValueError("no held-out tokens were evaluated")
        out: Dict[str, Any] = {
            "loss": self.loss_sum / self.tokens,
            "tokens": self.tokens,
            "buckets": {f"{lo}-{hi}": (self.bucket_sum[(lo, hi)] / self.bucket_tokens[(lo, hi)]
                                       if self.bucket_tokens[(lo, hi)] else None)
                        for lo, hi in BUCKETS},
        }
        if self.lambda_sum is not None:
            # a = exp(-mean lambda): the geometric mean of a at Delta = 1.
            a = np.exp(-self.lambda_sum / self.sequences)
            out["decay"] = {"min": a.min(axis=1).tolist(),
                            "median": np.median(a, axis=1).tolist(),
                            "max": a.max(axis=1).tolist()}
            out["state_rms"] = (self.rms_sum / self.sequences).tolist()
        return out


class SkipCounter:
    """Optimizer steps the fp16 loss scaler skipped, cumulative over the run.

    A skip shows as the scale dropping across scaler.update(). Checkpointed with
    the optimizer, so steps replayed after a kill are not counted twice;
    design 7.3's 1% bound is read off it.
    """

    def __init__(self) -> None:
        self.count = 0

    def observe(self, scale_before: float, scale_after: float) -> None:
        if scale_after < scale_before:
            self.count += 1

    def state_dict(self) -> Dict[str, int]:
        return {"count": self.count}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self.count = int(state["count"])
