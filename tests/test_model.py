"""The assembled model: its size, its starting loss, that it learns, and the
contracts the trainer and the evaluator call."""

import math
from pathlib import Path

import pytest
import torch

from deluge.config import ModelConfig, load_model_config
from deluge.model import build
from deluge.model.init import STD

REPO = Path(__file__).resolve().parents[1]
ARMS = ["A0_scan_only", "A1_conv_baseline", "A1_lru_baseline", "A0_transformer_baseline"]

TINY = dict(d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, window=16, chunk=8,
            summary_sources=[], vocab_size=128, tie_embeddings=True, n_units=1,
            n_sink=2, ffn={"type": "dense", "d_ff": 128}, mod=None, mtp=None)
TINY_MIXERS = {
    "cfc": {"type": "cfc", "d_inner": 48, "n_decay_heads": 4, "conv_kernel": 4,
            "delta_aware": True, "a_min": 0.9, "a_max": 0.999},
    "lru": {"type": "lru", "d_inner": 48, "n_decay_heads": 4, "a_min": 0.9, "a_max": 0.999},
    "gated_conv": {"type": "gated_conv", "d_inner": 48, "conv_kernel": 3},
    "attention": {"type": "attention"},
}


def tiny(kind: str) -> ModelConfig:
    return ModelConfig.from_dict({**TINY, "mixer": TINY_MIXERS[kind]})


def arm(name: str) -> ModelConfig:
    return load_model_config(REPO / f"configs/ablations/{name}.yaml")


@pytest.mark.parametrize("name", ARMS)
def test_the_built_model_has_exactly_the_configured_parameters(name):
    # The M1 param match is made in config.py; it only holds if the modules
    # are what config.py costs. On the meta device: no memory, no init time.
    cfg = arm(name)
    with torch.device("meta"):
        model = build(cfg)
    assert sum(p.numel() for p in model.parameters()) == cfg.params().total


@pytest.mark.parametrize("name", ARMS)
def test_a_fresh_model_starts_near_the_uniform_loss(name):
    # ln V + sigma^2 / 2, with sigma = STD * sqrt(d_model) ~ 0.55 the logit SD a
    # tied std-0.02 embedding gives through a unit-RMS final norm: ~10.53 nats.
    # Far off it means an init bug -- a gate that starts closed, a residual
    # branch that starts too loud -- not bad luck.
    cfg = arm(name)
    torch.manual_seed(0)
    model = build(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 129), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        loss, _ = model(tokens[:, :-1], tokens[:, 1:])
    expected = math.log(cfg.vocab_size) + (STD * math.sqrt(cfg.d_model)) ** 2 / 2
    assert abs(loss.item() - expected) < 0.3, f"{loss.item():.3f} against {expected:.3f}"


@pytest.mark.parametrize("kind", TINY_MIXERS)
def test_a_tiny_model_overfits_one_batch(kind):
    torch.manual_seed(0)
    model = build(tiny(kind))
    tokens = torch.randint(0, 128, (2, 33), generator=torch.Generator().manual_seed(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(150):
        loss, _ = model(tokens[:, :-1], tokens[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0], f"{losses[0]:.3f} -> {losses[-1]:.3f}"


def test_per_token_losses_average_to_the_training_loss():
    model = build(tiny("cfc"))
    tokens = torch.randint(0, 128, (3, 33))
    loss, metrics = model(tokens[:, :-1], tokens[:, 1:])
    losses, stats = model(tokens[:, :-1], tokens[:, 1:], reduction="none")
    assert loss.ndim == 0 and metrics == {}
    assert losses.shape == (3, 32)
    torch.testing.assert_close(losses.mean(), loss)
    # Leading batch dimension on everything, so DataParallel can gather it:
    # 3 recurrent layers (C, C, C, A) of 4 decay heads.
    assert stats["decay_lambda"].shape == (3, 3, 4)
    assert stats["state_rms"].shape == (3, 3)


def test_the_monitors_read_the_initial_decay_grid():
    # W_f = 0 at init, so the mean lambda is softplus(b_f) whatever the input.
    model = build(tiny("cfc"))
    tokens = torch.randint(0, 128, (2, 33))
    _, stats = model(tokens[:, :-1], tokens[:, 1:], reduction="none")
    expected = torch.nn.functional.softplus(model.blocks[0].mixer.W_f.bias.float())
    torch.testing.assert_close(stats["decay_lambda"][0, 0], expected)


@pytest.mark.parametrize("kind", ["gated_conv", "attention"])
def test_arms_without_recurrent_state_report_no_monitors(kind):
    model = build(tiny(kind))
    tokens = torch.randint(0, 128, (2, 17))
    _, stats = model(tokens[:, :-1], tokens[:, 1:], reduction="none")
    assert stats == {}


def test_rejects_an_unknown_reduction():
    model = build(tiny("cfc"))
    tokens = torch.randint(0, 128, (1, 9))
    with pytest.raises(ValueError, match="reduction"):
        model(tokens[:, :-1], tokens[:, 1:], reduction="sum")


def test_no_weight_decay_names_the_conv_buffers_and_the_sinks():
    names = build(tiny("cfc")).no_weight_decay()
    assert "blocks.0.mixer.conv_buf0" in names
    assert {"blocks.3.mixer.sink_k", "blocks.3.mixer.sink_v"} <= set(names)
    assert all(n.endswith(("conv_buf0", "sink_k", "sink_v")) for n in names)


def test_m2_features_are_refused_rather_than_ignored():
    # dev.yaml has chunk summaries on; building it would silently drop them.
    with pytest.raises(NotImplementedError, match="M2"):
        build(load_model_config(REPO / "configs/model/dev.yaml"))
