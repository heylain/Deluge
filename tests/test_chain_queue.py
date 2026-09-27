"""The chain's queue: a typo here should fail the suite, not a Kaggle session."""

import math
from pathlib import Path

import pytest

from deluge.chain.queue import load_queue
from deluge.config import ConfigError, load_model_config
from deluge.train.config import load_train_config

ROOT = Path(__file__).resolve().parents[1]

GOOD = """
runs:
  - name: a0-scan-only
    model: configs/model/dev.yaml
    train: configs/train/screen.yaml
    accelerator: gpu
    data: heylain/deluge-tokens:tokens.bin
"""


def write(tmp_path, text):
    path = tmp_path / "runs.yaml"
    path.write_text(text)
    return path


def test_loads_a_run_and_fills_the_default_factory(tmp_path):
    [run] = load_queue(write(tmp_path, GOOD))
    assert run["name"] == "a0-scan-only"
    assert run["model_impl"] == "deluge.model:build"


def test_an_empty_queue_is_valid(tmp_path):
    assert load_queue(write(tmp_path, "runs: []\n")) == []


@pytest.mark.parametrize("field, value, match", [
    ("name", "A0_scan", "name"),
    ("name", "x" * 42, "41"),
    ("accelerator", "tpu", "accelerator"),
    ("data", "tokens.bin", "data"),
])
def test_rejects_bad_fields(tmp_path, field, value, match):
    text = GOOD.replace({"name": "a0-scan-only", "accelerator": "gpu",
                         "data": "heylain/deluge-tokens:tokens.bin"}[field], value)
    with pytest.raises(ConfigError, match=match):
        load_queue(write(tmp_path, text))


def test_rejects_a_missing_field(tmp_path):
    with pytest.raises(ConfigError, match="missing data"):
        load_queue(write(tmp_path, GOOD.replace(
            "    data: heylain/deluge-tokens:tokens.bin\n", "")))


def test_rejects_an_unknown_field(tmp_path):
    with pytest.raises(ConfigError, match="unknown field.*epochs"):
        load_queue(write(tmp_path, GOOD + "    epochs: 3\n"))


def test_rejects_duplicate_names(tmp_path):
    body = GOOD.split("runs:\n")[1]
    with pytest.raises(ConfigError, match="more than once"):
        load_queue(write(tmp_path, "runs:\n" + body + body))


def test_committed_queue_loads_and_its_configs_parse():
    # Review focus 3: a wrong path in runs.yaml must fail here, not on Kaggle.
    for run in load_queue(ROOT / "configs/chain/runs.yaml"):
        load_model_config(ROOT / run["model"])
        load_train_config(ROOT / run["train"])


def test_smoke_budget_needs_exactly_three_sessions():
    # smoke.py's sleep sets the step time, so the session count is arithmetic.
    # Allow up to 0.2 s of real compute per step on Kaggle's CPU on top of it.
    from deluge.train.smoke import SECONDS_PER_STEP
    cfg = load_train_config(ROOT / "configs/train/smoke.yaml")
    assert cfg.grad_accum == 1, "one forward per step, so one sleep per step"
    window = cfg.deadline_minutes * 60
    for step_seconds in (SECONDS_PER_STEP, SECONDS_PER_STEP + 0.2):
        assert math.ceil(cfg.total_steps / (window // step_seconds)) == 3


def test_smoke_model_trains_and_the_crashing_arm_crashes(monkeypatch):
    torch = pytest.importorskip("torch")
    from deluge.train import smoke
    monkeypatch.setattr(smoke, "SECONDS_PER_STEP", 0.0)
    cfg = load_model_config(ROOT / "configs/model/screen.yaml")
    model = smoke.build(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    loss = model(x, x)
    loss.backward()
    assert loss.ndim == 0 and model.head.weight.grad is not None
    with pytest.raises(RuntimeError, match="deliberate"):
        smoke.build_crashing(cfg)
