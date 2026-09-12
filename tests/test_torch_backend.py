"""The torch adapter plugs into the harness sockets correctly.

The interesting property -- that resume is exact -- is proved in test_resume.py
without torch. What is left here is narrower and specific to the adapter: that a
real nn.Module and a real AdamW checkpoint and restore through the same
interface, that the loss scaler's state survives (the one piece of optimizer
state that is easy to forget), and that the model contract in train.py's
docstring is the one build_backend actually calls.

Skipped without torch, which is a [train] extra. Runs on Kaggle, where torch is
preinstalled. CPU and fp32 on purpose: fp16 on CUDA accumulates in a
nondeterministic order, so bit-equality is not the right assertion there.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn                                        # noqa: E402
import torch.nn.functional as F                              # noqa: E402

from deluge.config import load_model_config                   # noqa: E402
from deluge.data import TokenStream                           # noqa: E402
from deluge.train.checkpoint import Checkpointer              # noqa: E402
from deluge.train.config import TrainConfig                   # noqa: E402
from deluge.train.loop import TrainLoop                       # noqa: E402
from deluge.train.trainer import build_backend                # noqa: E402

REPO = Path(__file__).resolve().parents[1]

BUDGET = dict(
    seq_len=32, global_batch_tokens=256, micro_batch=2,
    total_tokens=256 * 8, lr=1e-2, min_lr_ratio=0.1, warmup_tokens=256,
    weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
    dtype="fp32", seed=5, checkpoint_every_tokens=256 * 4,
    keep_last=2, deadline_minutes=None,
)


class FakeClock:
    """One second per step, so a deadline stops the run at a known step."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TinyModel(nn.Module):
    """Stands in for M1: owns its objective, returns (loss, metrics)."""

    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, inputs, targets):
        logits = self.head(self.norm(self.embed(inputs)))
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, {"ce": loss.detach()}


@pytest.fixture
def model_cfg():
    # A real config, shrunk only where the stub model reads it.
    return dataclasses.replace(
        load_model_config(REPO / "configs/model/screen.yaml"),
        d_model=64, vocab_size=512, n_heads=1, head_dim=64)


def make_loop(tmp_path, cfg, model_cfg, device="cpu"):
    clock = FakeClock()
    torch.manual_seed(0)                 # identical init in every process
    step_fn, stateful = build_backend(
        model_cfg, cfg, device=device, data_parallel=False, factory=TinyModel)

    def timed(batches, lr):
        clock.t += 1.0
        return step_fn(batches, lr)

    tokens = np.random.default_rng(0).integers(
        0, model_cfg.vocab_size, size=20_000, dtype=np.uint16)
    loop = TrainLoop(
        cfg=cfg,
        stream=TokenStream(tokens, cfg.seq_len, cfg.seed),
        checkpointer=Checkpointer(tmp_path, keep_last=cfg.keep_last),
        stateful=stateful,
        step_fn=timed,
        meta={"train_fingerprint": cfg.fingerprint, "model_fingerprint": "tiny"},
        clock=clock,
    )
    return loop, stateful


def test_a_torch_model_and_adamw_resume_bit_identically(tmp_path, model_cfg):
    cfg = TrainConfig(**BUDGET)

    straight, state_a = make_loop(tmp_path / "a", cfg, model_cfg)
    assert straight.run().step == 8

    # Same budget -- the LR schedule must not change -- with a deadline that
    # trips after 3 steps at one second each.
    interrupted = dataclasses.replace(cfg, deadline_minutes=4 / 60)
    first, _ = make_loop(tmp_path / "b", interrupted, model_cfg)
    assert first.run().step == 3

    second, state_b = make_loop(tmp_path / "b", cfg, model_cfg)
    assert second.run().step == 8

    for name, tensor in state_a["model"].state_dict().items():
        assert torch.equal(tensor, state_b["model"].state_dict()[name]), name


def test_adamw_moments_survive_the_restart(tmp_path, model_cfg):
    # Adam's second moment is what a naive "reload the weights" resume loses,
    # and losing it shows up as a visible loss spike rather than an error.
    cfg = TrainConfig(**BUDGET)
    straight, state_a = make_loop(tmp_path / "m", cfg, model_cfg)
    straight.run()

    interrupted = dataclasses.replace(cfg, deadline_minutes=4 / 60)
    first, _ = make_loop(tmp_path / "n", interrupted, model_cfg)
    first.run()
    second, state_b = make_loop(tmp_path / "n", cfg, model_cfg)
    second.run()

    a = state_a["optimizer"].state_dict()["state"]
    b = state_b["optimizer"].state_dict()["state"]
    assert set(a) == set(b)
    for index in a:
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(torch.as_tensor(a[index][key]),
                               torch.as_tensor(b[index][key])), f"{index}.{key}"


def test_the_loss_scaler_is_checkpointed(tmp_path, model_cfg):
    cfg = TrainConfig(**BUDGET)
    _, stateful = make_loop(tmp_path / "s", cfg, model_cfg)
    assert "scaler" in stateful


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_an_enabled_scaler_carries_its_scale(tmp_path, model_cfg):
    # The scale ramps over the run; resuming with it reset replays the ramp and
    # can overflow the first steps back.
    cfg = dataclasses.replace(TrainConfig(**BUDGET), dtype="fp16")
    _, stateful = make_loop(tmp_path / "g", cfg, model_cfg, device="cuda")
    assert "scale" in stateful["scaler"].state_dict()


def test_an_empty_scaler_state_loads_without_raising(tmp_path, model_cfg):
    # A run smoke-tested on CPU (scaler disabled, empty state) must still be
    # resumable on a GPU, where the scaler is enabled.
    cfg = TrainConfig(**BUDGET)
    _, stateful = make_loop(tmp_path / "e", cfg, model_cfg)
    stateful["scaler"].load_state_dict({})


def test_checkpoints_hold_an_unwrapped_model(tmp_path, model_cfg):
    # DataParallel prefixes every key with "module."; the checkpoint must not
    # carry it, or a 2-GPU session cannot be resumed on 1.
    cfg = TrainConfig(**BUDGET)
    loop, stateful = make_loop(tmp_path / "u", cfg, model_cfg)
    loop.run()
    payload = Checkpointer(tmp_path / "u").load_latest(stateful)
    assert not any(k.startswith("module.") for k in payload["state"]["model"])


def test_weight_decay_skips_one_dimensional_parameters(tmp_path, model_cfg):
    cfg = TrainConfig(**BUDGET)
    _, stateful = make_loop(tmp_path / "d", cfg, model_cfg)
    decayed, undecayed = stateful["optimizer"].param_groups
    assert decayed["weight_decay"] == cfg.weight_decay
    assert undecayed["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decayed["params"])
    assert all(p.ndim < 2 for p in undecayed["params"])


def test_the_documented_model_contract_is_the_one_that_is_called(model_cfg):
    # train.py's docstring promises model(inputs, targets) -> loss | (loss, metrics)
    # with int64 [batch, seq_len] tensors. Assert it rather than trusting it.
    seen = {}

    class ContractModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(2, 2))

        def forward(self, inputs, targets):
            seen["shapes"] = (tuple(inputs.shape), tuple(targets.shape))
            seen["dtypes"] = (inputs.dtype, targets.dtype)
            return self.w.sum() + inputs.float().mean(), {}

    cfg = TrainConfig(**BUDGET)
    step_fn, _ = build_backend(model_cfg, cfg, "cpu", False, lambda c: ContractModel())
    step_fn([(np.zeros((2, 32), dtype=np.int64),
              np.ones((2, 32), dtype=np.int64))], lr=1e-3)

    assert seen["shapes"] == ((2, 32), (2, 32))
    assert seen["dtypes"] == (torch.int64, torch.int64)


def test_a_bare_loss_return_is_accepted(model_cfg):
    class BareLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(4, 4))

        def forward(self, inputs, targets):
            return (self.w ** 2).sum()

    cfg = TrainConfig(**BUDGET)
    step_fn, _ = build_backend(model_cfg, cfg, "cpu", False, lambda c: BareLoss())
    out = step_fn([(np.zeros((2, 32), dtype=np.int64),
                    np.zeros((2, 32), dtype=np.int64))], lr=1e-3)
    assert "loss" in out and out["loss"] > 0


def test_extra_loss_terms_reach_the_run_log(model_cfg):
    # Spec 7's objective has three terms; a log that only records the sum
    # cannot say which one moved.
    class ThreeTerms(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(4, 4))

        def forward(self, inputs, targets):
            total = (self.w ** 2).sum()
            return total, {"ce": total.detach() * 0.9,
                           "mtp": total.detach() * 0.1}

    cfg = TrainConfig(**BUDGET)
    step_fn, _ = build_backend(model_cfg, cfg, "cpu", False, lambda c: ThreeTerms())
    out = step_fn([(np.zeros((2, 32), dtype=np.int64),
                    np.zeros((2, 32), dtype=np.int64))], lr=1e-3)
    assert {"loss", "ce", "mtp", "grad_norm"} <= set(out)
