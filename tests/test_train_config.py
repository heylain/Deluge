"""Training budget invariants, in the style of test_config_invariants.py."""

import dataclasses
from pathlib import Path

import pytest

from deluge.config import ConfigError
from deluge.train.config import TrainConfig, load_train_config

REPO = Path(__file__).resolve().parents[1]

SCREEN = dict(
    seq_len=1024,
    global_batch_tokens=262144,
    micro_batch=8,
    total_tokens=500_000_000,
    lr=3.0e-4,
    min_lr_ratio=0.1,
    warmup_tokens=10_000_000,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    mtp_weight=0.3,
    mtp_weight_final=0.1,
    mtp_anneal_from=0.6,
    mod_pred_weight=0.01,
    router_z_weight=1.0e-3,
    dtype="fp16",
    seed=1234,
    checkpoint_every_tokens=25_000_000,
    keep_last=2,
    deadline_minutes=660,
)


def make(**overrides) -> TrainConfig:
    return TrainConfig(**{**SCREEN, **overrides})


def test_accepts_the_screening_budget():
    assert make().grad_accum == 32


def test_rejects_a_global_batch_that_is_not_whole_micro_batches():
    # A ragged tail micro-batch makes the effective batch differ from the
    # configured one, differently on every GPU the run touches.
    with pytest.raises(ConfigError, match="whole number of micro-batches"):
        make(micro_batch=7)


def test_rejects_an_unknown_dtype():
    with pytest.raises(ConfigError, match="dtype"):
        make(dtype="fp8")


def test_rejects_warmup_longer_than_the_run():
    with pytest.raises(ConfigError, match="warmup_tokens"):
        make(warmup_tokens=500_000_000)


def test_rejects_a_budget_shorter_than_one_step():
    with pytest.raises(ConfigError, match="less than one optimizer step"):
        make(total_tokens=1000)


def test_rejects_min_lr_ratio_outside_zero_to_one():
    # It is a fraction of lr; 3e-5 passed here would mean a floor of 3e-5 * lr.
    with pytest.raises(ConfigError, match="min_lr_ratio"):
        make(min_lr_ratio=3.0)


def test_rejects_keep_last_below_one():
    with pytest.raises(ConfigError, match="keep_last"):
        make(keep_last=0)


def test_rejects_a_non_positive_deadline():
    with pytest.raises(ConfigError, match="deadline_minutes"):
        make(deadline_minutes=0)


def test_allows_no_deadline():
    assert make(deadline_minutes=None).deadline_minutes is None


@pytest.mark.parametrize("field", ["beta1", "beta2"])
def test_rejects_betas_outside_the_unit_interval(field):
    with pytest.raises(ConfigError, match=field):
        make(**{field: 1.0})


def test_derived_step_counts():
    cfg = make()
    assert cfg.sequences_per_step == 256
    assert cfg.total_steps == 500_000_000 // 262144
    assert cfg.checkpoint_every_steps == 25_000_000 // 262144


def test_checkpoint_cadence_is_never_zero_steps():
    # A cadence finer than one step would save every step, not never.
    assert make(checkpoint_every_tokens=1).checkpoint_every_steps == 1


# ---- schedule ------------------------------------------------------------- #

def test_lr_warms_up_linearly_then_decays_to_the_floor():
    cfg = make()
    assert cfg.lr_at(0) == 0
    assert cfg.lr_at(cfg.warmup_tokens // 2) == pytest.approx(cfg.lr / 2)
    assert cfg.lr_at(cfg.warmup_tokens) == pytest.approx(cfg.lr)
    assert cfg.lr_at(cfg.total_tokens) == pytest.approx(cfg.lr * cfg.min_lr_ratio)


def test_lr_is_monotonic_after_warmup():
    cfg = make()
    span = range(cfg.warmup_tokens, cfg.total_tokens, cfg.total_tokens // 50)
    values = [cfg.lr_at(t) for t in span]
    assert values == sorted(values, reverse=True)


def test_lr_does_not_run_past_the_floor_if_the_budget_is_overrun():
    cfg = make()
    assert cfg.lr_at(cfg.total_tokens * 2) == pytest.approx(cfg.lr * cfg.min_lr_ratio)


# ---- fingerprint ---------------------------------------------------------- #

def test_fingerprint_ignores_how_the_run_is_hosted():
    # micro_batch, cadence and deadline are how a run fits its machine; changing
    # them between sessions must not block a resume.
    cfg = make()
    for field, value in [("micro_batch", 4), ("keep_last", 5),
                         ("deadline_minutes", 120),
                         ("checkpoint_every_tokens", 1_000_000)]:
        assert dataclasses.replace(cfg, **{field: value}).fingerprint == cfg.fingerprint


@pytest.mark.parametrize("field,value", [
    ("lr", 1e-3), ("seed", 5), ("total_tokens", 600_000_000),
    ("seq_len", 512), ("global_batch_tokens", 524288), ("dtype", "bf16"),
])
def test_fingerprint_catches_a_changed_optimization(field, value):
    cfg = make()
    assert dataclasses.replace(cfg, **{field: value}).fingerprint != cfg.fingerprint


# ---- the shipped configs -------------------------------------------------- #

def test_shipped_budgets_load():
    for name in ("base", "screen", "dev"):
        assert load_train_config(REPO / f"configs/train/{name}.yaml").total_steps > 0


def test_dev_budget_is_the_twenty_tokens_per_param_floor():
    # Spec 7 sizes the dev budget at ~20 tokens/param. The param count comes
    # from config.py, so this fails if the dev architecture moves and the
    # budget is not moved with it.
    from deluge.config import load_model_config

    params = load_model_config(REPO / "configs/model/dev.yaml").params().total
    budget = load_train_config(REPO / "configs/train/dev.yaml").total_tokens
    assert 19 < budget / params < 21, (
        f"dev budget is {budget / params:.1f} tokens/param, not ~20")


def test_screening_is_a_measurement_not_a_model():
    # Screening is deliberately far under the floor: it ranks arms, it does not
    # train one. If this ever approaches 20 the two budgets have collapsed into
    # one and the cheap rung of the ladder is gone.
    from deluge.config import load_model_config

    params = load_model_config(REPO / "configs/model/dev.yaml").params().total
    budget = load_train_config(REPO / "configs/train/screen.yaml").total_tokens
    assert budget / params < 5


def test_screen_and_dev_differ_only_in_the_budget():
    screen = load_train_config(REPO / "configs/train/screen.yaml")
    dev = load_train_config(REPO / "configs/train/dev.yaml")
    budget_fields = {"total_tokens", "warmup_tokens", "checkpoint_every_tokens"}
    for f in dataclasses.fields(TrainConfig):
        if f.name not in budget_fields:
            assert getattr(screen, f.name) == getattr(dev, f.name), (
                f"{f.name} differs between the screening and dev budgets; "
                f"arms trained under the two would not be comparable")


# ---- auxiliary loss weights (spec 7) -------------------------------------- #

@pytest.mark.parametrize("field", ["mtp_weight", "mtp_weight_final",
                                   "mod_pred_weight", "router_z_weight"])
def test_rejects_a_negative_loss_weight(field):
    # 0 is how a term is disabled; negative would reward the thing it penalises.
    with pytest.raises(ConfigError, match=field):
        make(**{field: -0.1})


@pytest.mark.parametrize("field", ["mtp_weight", "mod_pred_weight",
                                   "router_z_weight"])
def test_a_loss_weight_may_be_zero(field):
    assert getattr(make(**{field: 0.0}), field) == 0.0


def test_rejects_an_anneal_point_outside_the_run():
    with pytest.raises(ConfigError, match="mtp_anneal_from"):
        make(mtp_anneal_from=1.5)


def test_mtp_weight_is_flat_then_anneals_to_the_floor():
    # Spec 7: 0.3, annealed to 0.1 after 60% of training.
    cfg = make()
    start = int(cfg.mtp_anneal_from * cfg.total_tokens)
    assert cfg.mtp_weight_at(0) == pytest.approx(cfg.mtp_weight)
    assert cfg.mtp_weight_at(start) == pytest.approx(cfg.mtp_weight)
    assert cfg.mtp_weight_at(cfg.total_tokens) == pytest.approx(cfg.mtp_weight_final)


def test_mtp_weight_anneals_smoothly_rather_than_stepping():
    # A step two thirds of the way through a run puts a discontinuity exactly
    # where a loss spike is hardest to attribute. Halfway through the anneal
    # should be halfway between the two weights.
    cfg = make()
    start = cfg.mtp_anneal_from * cfg.total_tokens
    midpoint = int(start + (cfg.total_tokens - start) / 2)
    assert cfg.mtp_weight_at(midpoint) == pytest.approx(
        (cfg.mtp_weight + cfg.mtp_weight_final) / 2)


def test_mtp_weight_does_not_run_past_the_floor_if_the_budget_is_overrun():
    cfg = make()
    assert cfg.mtp_weight_at(cfg.total_tokens * 2) == pytest.approx(cfg.mtp_weight_final)


def test_mtp_weight_is_monotonic():
    cfg = make()
    span = range(0, cfg.total_tokens, cfg.total_tokens // 50)
    weights = [cfg.mtp_weight_at(t) for t in span]
    assert weights == sorted(weights, reverse=True)


def test_an_anneal_from_one_never_anneals():
    cfg = make(mtp_anneal_from=1.0)
    assert cfg.mtp_weight_at(cfg.total_tokens) == pytest.approx(cfg.mtp_weight)


@pytest.mark.parametrize("field,value", [
    ("mtp_weight", 0.5), ("mtp_weight_final", 0.2), ("mtp_anneal_from", 0.8),
    ("mod_pred_weight", 0.02), ("router_z_weight", 1e-2),
])
def test_the_fingerprint_catches_a_changed_objective(field, value):
    # Loss weights define the optimization as much as the LR does; resuming
    # under different ones would train one run's weights on another's objective.
    cfg = make()
    assert dataclasses.replace(cfg, **{field: value}).fingerprint != cfg.fingerprint


def test_the_dev_scale_peak_lr_follows_the_spec():
    # Spec 7: peak 3e-4, dev 6e-4. Everything extending train/base.yaml is
    # dev-scale, so base carries 6e-4 and a target budget overrides it.
    assert load_train_config(REPO / "configs/train/screen.yaml").lr == pytest.approx(6e-4)
