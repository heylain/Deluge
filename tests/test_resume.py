"""Resume must be exact, not approximate.

The property under test: training N steps in one process and training N steps
across a kill and a restart produce bit-identical weights. That is the whole
reason the harness exists -- a run split over three Kaggle sessions has to be
the same run, not three runs stitched together.

The model here is a numpy stub, on purpose. Everything the property depends on
(data position, RNG, optimizer state, LR schedule, step accounting) lives in
the harness, so the property is provable without torch, without a GPU, and in
under a second. tests/test_torch_backend.py checks the much smaller claim that
torch objects plug into the same sockets.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from deluge.data import TokenStream
from deluge.train.checkpoint import Checkpointer
from deluge.train.config import TrainConfig
from deluge.train.loop import ResumeMismatch, TrainLoop

WIDTH = 8

BUDGET = dict(
    seq_len=16,
    global_batch_tokens=128,      # 8 sequences per step
    micro_batch=2,                # -> grad_accum 4
    total_tokens=128 * 20,        # 20 steps
    lr=0.1,
    min_lr_ratio=0.1,
    warmup_tokens=128 * 3,
    weight_decay=0.0,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    mtp_weight=0.3,
    mtp_weight_final=0.1,
    mtp_anneal_from=0.6,
    mod_pred_weight=0.01,
    router_z_weight=1.0e-3,
    dtype="fp32",
    seed=99,
    checkpoint_every_tokens=128 * 4,
    keep_last=2,
    deadline_minutes=None,
)


class FakeClock:
    """Wall time the test advances by hand, so deadlines are deterministic."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class StubModel:
    def __init__(self):
        self.w = np.zeros(WIDTH, dtype=np.float64)

    def state_dict(self):
        return {"w": self.w.copy()}

    def load_state_dict(self, state):
        self.w[:] = state["w"]


class StubOptimizer:
    """Momentum SGD. Momentum is state that a naive resume silently loses."""

    def __init__(self):
        self.m = np.zeros(WIDTH, dtype=np.float64)

    def state_dict(self):
        return {"m": self.m.copy()}

    def load_state_dict(self, state):
        self.m[:] = state["m"]


def make_step_fn(model, optimizer, clock, losses, seconds_per_step=1.0):
    """A step that is sensitive to every piece of state a resume must restore."""

    def step_fn(batches, lr):
        clock.t += seconds_per_step
        # Gradient from the actual tokens: a wrong data position changes it.
        grad = np.zeros(WIDTH)
        for inputs, targets in batches:
            grad += np.bincount(inputs.ravel() % WIDTH, minlength=WIDTH)
            grad += np.bincount(targets.ravel() % WIDTH, minlength=WIDTH)
        grad /= grad.sum()
        # Noise from the global RNG: a wrong RNG restore changes it.
        grad += 0.01 * np.random.randn(WIDTH)
        optimizer.m[:] = 0.9 * optimizer.m + grad
        model.w -= lr * optimizer.m
        loss = float(np.abs(model.w).sum())
        losses.append((lr, loss))
        return loss

    return step_fn


def build(tmp_path: Path, cfg: TrainConfig, clock=None, corpus_seed=0):
    clock = clock or FakeClock()
    model, optimizer, losses = StubModel(), StubOptimizer(), []
    tokens = np.random.default_rng(corpus_seed).integers(
        0, 32000, size=20_000, dtype=np.uint16)
    stream = TokenStream(tokens, cfg.seq_len, cfg.seed)
    loop = TrainLoop(
        cfg=cfg,
        stream=stream,
        checkpointer=Checkpointer(tmp_path, keep_last=cfg.keep_last),
        stateful={"model": model, "optimizer": optimizer},
        step_fn=make_step_fn(model, optimizer, clock, losses),
        meta={"train_fingerprint": cfg.fingerprint, "model_fingerprint": "stub"},
        clock=clock,
    )
    return loop, model, optimizer, losses, clock


# --------------------------------------------------------------------------- #
# The central property
# --------------------------------------------------------------------------- #

def test_resuming_a_killed_run_is_bit_identical_to_never_stopping(tmp_path):
    cfg = TrainConfig(**BUDGET)

    straight, model_a, _, losses_a, _ = build(tmp_path / "a", cfg)
    outcome_a = straight.run()
    assert outcome_a.complete and outcome_a.step == 20

    # Same budget, but a deadline that trips partway. deadline_minutes is in
    # minutes and the fake clock ticks one second per step, so 10/60 stops the
    # loop once the *next* step would cross 10 s -- after step 9.
    interrupted_cfg = dataclasses.replace(cfg, deadline_minutes=10 / 60)
    first, _, _, losses_first, _ = build(tmp_path / "b", interrupted_cfg)
    outcome_first = first.run()
    assert not outcome_first.complete
    assert outcome_first.reason == "deadline"
    assert outcome_first.step == 9

    # A genuinely fresh process: new model, new optimizer, new RNG, new loop.
    second, model_b, _, losses_second, _ = build(tmp_path / "b", cfg)
    outcome_b = second.run()

    assert outcome_b.complete and outcome_b.step == 20
    assert np.array_equal(model_a.w, model_b.w), (
        f"resumed weights differ: {model_a.w} vs {model_b.w}")
    assert losses_a == losses_first + losses_second


def test_resume_picks_up_the_lr_schedule_where_it_left_off(tmp_path):
    # The schedule is a function of tokens_seen, so this is really a check that
    # tokens_seen survives the restart rather than restarting from warmup.
    cfg = TrainConfig(**BUDGET)
    interrupted = dataclasses.replace(cfg, deadline_minutes=10 / 60)

    first, _, _, losses_first, _ = build(tmp_path / "r", interrupted)
    first.run()
    second, _, _, losses_second, _ = build(tmp_path / "r", cfg)
    second.run()

    assert losses_second[0][0] == pytest.approx(cfg.lr_at(9 * cfg.global_batch_tokens))
    assert losses_first[0][0] == pytest.approx(cfg.lr_at(0))


def test_a_completed_run_resumes_to_a_no_op(tmp_path):
    cfg = TrainConfig(**BUDGET)
    first, model_a, _, _, _ = build(tmp_path / "c", cfg)
    first.run()
    again, model_b, _, losses, _ = build(tmp_path / "c", cfg)
    outcome = again.run()

    assert outcome.complete and outcome.step == cfg.total_steps
    assert losses == [], "a finished run took further steps on restart"
    assert np.array_equal(model_a.w, model_b.w)


# --------------------------------------------------------------------------- #
# Batch shape is a host detail, not an optimization one
# --------------------------------------------------------------------------- #

def test_micro_batch_does_not_change_which_sequences_a_step_sees(tmp_path):
    # This is what lets the same config train on a 16 GB T4 and an 80 GB card.
    cfg = TrainConfig(**BUDGET)
    wide = dataclasses.replace(cfg, micro_batch=8)

    narrow_loop, *_ = build(tmp_path / "n", cfg)
    wide_loop, *_ = build(tmp_path / "w", wide)

    assert cfg.grad_accum == 4 and wide.grad_accum == 1
    for step in (0, 1, 7):
        narrow = np.concatenate([b[0] for b in narrow_loop.batches_for_step(step)])
        broad = np.concatenate([b[0] for b in wide_loop.batches_for_step(step)])
        assert np.array_equal(narrow, broad)


def test_changing_micro_batch_mid_run_still_resumes_identically(tmp_path):
    cfg = TrainConfig(**BUDGET)
    interrupted = dataclasses.replace(cfg, deadline_minutes=10 / 60)

    straight, model_a, _, _, _ = build(tmp_path / "x", cfg)
    straight.run()

    first, _, _, _, _ = build(tmp_path / "y", interrupted)
    first.run()
    # Second session comes back on a host with more VRAM.
    second, model_b, _, _, _ = build(tmp_path / "y", dataclasses.replace(cfg, micro_batch=8))
    second.run()

    assert np.array_equal(model_a.w, model_b.w)


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

def test_resuming_under_a_different_budget_is_refused(tmp_path):
    cfg = TrainConfig(**BUDGET)
    interrupted = dataclasses.replace(cfg, deadline_minutes=10 / 60)
    first, *_ = build(tmp_path / "z", interrupted)
    first.run()

    other = dataclasses.replace(cfg, lr=0.5)
    second, *_ = build(tmp_path / "z", other)
    with pytest.raises(ResumeMismatch, match="train_fingerprint"):
        second.run()


def test_resuming_a_different_model_is_refused(tmp_path):
    cfg = TrainConfig(**BUDGET)
    interrupted = dataclasses.replace(cfg, deadline_minutes=10 / 60)
    first, *_ = build(tmp_path / "m", interrupted)
    first.run()

    second, *_ = build(tmp_path / "m", cfg)
    second.meta["model_fingerprint"] = "a different architecture"
    with pytest.raises(ResumeMismatch, match="model_fingerprint"):
        second.run()


def test_interrupted_run_reports_a_resumable_exit_code(tmp_path):
    cfg = dataclasses.replace(TrainConfig(**BUDGET), deadline_minutes=10 / 60)
    loop, *_ = build(tmp_path / "e", cfg)
    outcome = loop.run()
    assert outcome.exit_code == 2

    done, *_ = build(tmp_path / "e", TrainConfig(**BUDGET))
    assert done.run().exit_code == 0


def test_stopping_always_leaves_a_checkpoint_at_the_stopping_point(tmp_path):
    # checkpoint_every_steps is 4 here, so a stop at step 9 is mid-cadence: the
    # deadline save is what keeps steps 8 and 9 from being lost.
    cfg = dataclasses.replace(TrainConfig(**BUDGET), deadline_minutes=10 / 60)
    loop, *_ = build(tmp_path / "s", cfg)
    outcome = loop.run()

    assert outcome.step == 9
    assert outcome.checkpoint is not None
    assert outcome.checkpoint.name == "step-000000009.pt"


# --------------------------------------------------------------------------- #
# The step contract
# --------------------------------------------------------------------------- #

def test_extra_loss_terms_are_logged_alongside_the_total(tmp_path):
    # Spec 7's objective is L_ce + 0.3 L_mtp + 0.01 L_mod_pred. A log that only
    # records the sum cannot say which term moved.
    cfg = TrainConfig(**BUDGET)
    loop, *_ = build(tmp_path / "metrics", cfg)
    records = []
    loop.log_fn = records.append
    loop.step_fn = lambda batches, lr: {"loss": 1.0, "ce": 0.9, "mtp": 0.1}
    loop.run()

    steps = [r for r in records if r["event"] == "step"]
    assert len(steps) == cfg.total_steps
    assert steps[0]["loss"] == 1.0
    assert steps[0]["ce"] == 0.9 and steps[0]["mtp"] == 0.1


def test_a_step_that_reports_no_loss_fails_loudly(tmp_path):
    cfg = TrainConfig(**BUDGET)
    loop, *_ = build(tmp_path / "noloss", cfg)
    loop.step_fn = lambda batches, lr: {"ce": 0.5}
    with pytest.raises(KeyError, match="without a 'loss' key"):
        loop.run()
