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
    mtp_weight=0.3, mtp_weight_final=0.1, mtp_anneal_from=0.6,
    mod_pred_weight=0.01, router_z_weight=1.0e-3,
    dtype="fp32", seed=5, checkpoint_every_tokens=256 * 4,
    eval_every_tokens=None, keep_last=2, deadline_minutes=None,
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
    # A real config, shrunk only where the stub model reads it. Still has to
    # satisfy every invariant -- n_heads % n_kv_heads for GQA, n_heads *
    # head_dim == d_model -- which is why it is built through replace() on a
    # shipped config rather than assembled by hand.
    return dataclasses.replace(
        load_model_config(REPO / "configs/model/screen.yaml"),
        d_model=64, vocab_size=512, n_heads=1, n_kv_heads=1, head_dim=64)


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



# --------------------------------------------------------------------------- #
# M1: the model may name parameters to leave undecayed
# --------------------------------------------------------------------------- #

class WithState(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 4))
        self.buffer = nn.Parameter(torch.zeros(3, 4))       # 2-D, but learned state

    def no_weight_decay(self):
        return ["buffer"]

    def forward(self, inputs, targets):
        return (self.weight ** 2).sum() + self.buffer.sum()


def test_no_weight_decay_names_land_in_the_undecayed_group(model_cfg):
    model = WithState()
    _, stateful = build_backend(model_cfg, TrainConfig(**BUDGET), "cpu", False,
                                lambda c: model)
    decayed, undecayed = stateful["optimizer"].param_groups
    assert any(p is model.buffer for p in undecayed["params"])
    assert not any(p is model.buffer for p in decayed["params"])
    assert any(p is model.weight for p in decayed["params"])


def test_a_no_weight_decay_name_that_is_not_a_parameter_is_refused(model_cfg):
    class Typo(WithState):
        def no_weight_decay(self):
            return ["bufer"]

    with pytest.raises(ValueError, match="bufer"):
        build_backend(model_cfg, TrainConfig(**BUDGET), "cpu", False, lambda c: Typo())


def test_the_init_follows_the_seed(model_cfg):
    # Design 8.1: another seed changes the init as well as the data order. The
    # model used to be built before anything seeded torch, so its initial weights
    # came from whatever state the process's generator was in. So: one seed from
    # two different ambient states agrees, and another seed, from the first of
    # them, does not.
    cfg = TrainConfig(**BUDGET)

    def initial_weights(run_cfg, ambient_seed):
        torch.manual_seed(ambient_seed)
        _, stateful = build_backend(model_cfg, run_cfg, "cpu", False, TinyModel)
        return stateful["model"].state_dict()

    first = initial_weights(cfg, ambient_seed=123)
    second = initial_weights(cfg, ambient_seed=456)
    for name, tensor in first.items():
        assert torch.equal(tensor, second[name]), name

    other = initial_weights(dataclasses.replace(cfg, seed=cfg.seed + 1), ambient_seed=123)
    assert any(not torch.equal(tensor, other[name]) for name, tensor in first.items())


# --------------------------------------------------------------------------- #
# M1: the held-out evaluator, the skip counter, and a whole CLI run on CPU
# --------------------------------------------------------------------------- #

import json                                                   # noqa: E402

import yaml                                                   # noqa: E402

from deluge.config import ModelConfig                          # noqa: E402
from deluge.model import build                                 # noqa: E402
from deluge.train.trainer import build_evaluator, main         # noqa: E402

TINY_MODEL = dict(
    d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, window=16, chunk=8,
    summary_sources=[], vocab_size=128, tie_embeddings=True, n_units=1, n_sink=2,
    ffn={"type": "dense", "d_ff": 128}, mod=None, mtp=None,
    mixer={"type": "cfc", "d_inner": 48, "n_decay_heads": 4, "conv_kernel": 4,
           "delta_aware": True, "a_min": 0.9, "a_max": 0.999})


def test_the_skip_count_is_checkpointed(tmp_path, model_cfg):
    _, stateful = make_loop(tmp_path / "k", TrainConfig(**BUDGET), model_cfg)
    assert stateful["skips"].state_dict() == {"count": 0}


def heldout_pieces(n, piece_len, seed=0):
    return np.random.default_rng(seed).integers(0, 128, size=n * piece_len, dtype=np.uint16)


def test_the_evaluator_reads_every_piece_including_a_ragged_last_batch():
    # Review focus: 5 pieces at micro_batch 2 leaves a last batch of one.
    cfg = TrainConfig(**BUDGET)
    model = build(ModelConfig.from_dict(TINY_MODEL))
    evaluate = build_evaluator(model, cfg, "cpu", False, heldout_pieces(5, 33), 33)
    full, subset = evaluate(None)
    assert full["tokens"] == 5 * 32 and subset["tokens"] == 5 * 32
    assert len(full["decay"]["min"]) == 3 and len(full["state_rms"]) == 3
    assert model.training, "the evaluator must hand the model back in train mode"
    first_two, _ = evaluate(2)
    assert first_two["tokens"] == 2 * 32


def test_the_evaluator_passes_nothing_by_keyword(monkeypatch):
    # Review focus, on two GPUs: DataParallel splits tensors across the GPUs
    # but copies keyword arguments to every one, so a last batch of one piece
    # reaches the second GPU's replica as keywords with no inputs -- a
    # TypeError in the final eval. Positional arguments split with the batch.
    # No CPU test can run DataParallel's scatter, so this pins the call.
    calls = []

    class Recording(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            calls.append(kwargs)
            return self.module(*args, **kwargs)

    monkeypatch.setattr(torch.nn, "DataParallel", Recording)
    model = build(ModelConfig.from_dict(TINY_MODEL))
    evaluate = build_evaluator(model, TrainConfig(**BUDGET), "cpu", True,
                               heldout_pieces(5, 33), 33)
    full, _ = evaluate(None)
    assert full["tokens"] == 5 * 32
    assert calls == [{}, {}, {}]            # batches of 2, 2 and 1


def test_evals_leave_training_bit_identical(tmp_path):
    # Design 7.1: an eval touches neither the training RNG nor the data
    # position, so a run that evaluates every step trains the same weights.
    cfg = dataclasses.replace(TrainConfig(**BUDGET), total_tokens=256 * 4)
    model_cfg = ModelConfig.from_dict(TINY_MODEL)

    def train(directory, every):
        torch.manual_seed(0)
        step_fn, stateful = build_backend(model_cfg, cfg, "cpu", False, build)
        evaluate = build_evaluator(stateful["model"], cfg, "cpu", False,
                                   heldout_pieces(3, cfg.seq_len + 1, seed=1), cfg.seq_len + 1)
        tokens = np.random.default_rng(0).integers(0, 128, size=20_000, dtype=np.uint16)
        TrainLoop(cfg=dataclasses.replace(cfg, eval_every_tokens=every),
                  stream=TokenStream(tokens, cfg.seq_len, cfg.seed),
                  checkpointer=Checkpointer(directory), stateful=stateful,
                  step_fn=step_fn, clock=FakeClock(),
                  eval_fn=(lambda: evaluate(2)[0]) if every else None).run()
        return stateful

    plain = train(tmp_path / "plain", None)
    evaluated = train(tmp_path / "evals", cfg.global_batch_tokens)
    for name, tensor in plain["model"].state_dict().items():
        assert torch.equal(tensor, evaluated["model"].state_dict()[name]), name
    a = plain["optimizer"].state_dict()["state"]
    b = evaluated["optimizer"].state_dict()["state"]
    for index in a:
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(a[index][key], b[index][key]), f"{index}.{key}"


def write_run_inputs(tmp_path, train_overrides=None, piece_len=33, heldout=True,
                     provenance=None):
    """A tiny model config, a 4-step budget, and data in pieces of piece_len."""
    data = tmp_path / "data"
    data.mkdir()
    heldout_pieces(40, piece_len).tofile(data / "train.bin")
    if heldout:
        heldout_pieces(3, piece_len, seed=1).tofile(data / "heldout.bin")
    (data / "provenance.json").write_text(json.dumps({"piece_len": piece_len,
                                                      **(provenance or {})}))
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(TINY_MODEL))
    budget = {**BUDGET, "total_tokens": 256 * 4, "checkpoint_every_tokens": 256 * 2,
              "eval_every_tokens": 256 * 2, **(train_overrides or {})}
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(budget))
    return ["--model", str(tmp_path / "model.yaml"), "--train", str(tmp_path / "train.yaml"),
            "--data", str(data / "train.bin"), "--out", str(tmp_path / "run"),
            "--device", "cpu"]


def test_a_run_with_a_heldout_set_evaluates_and_finishes(tmp_path):
    argv = write_run_inputs(tmp_path)
    assert main(argv) == 0
    run = tmp_path / "run"
    final = json.loads((run / "final_eval.json").read_text())
    assert final["step"] == 4 and final["skipped_steps"] == 0
    assert final["full_tokens"] == 3 * 32 and final["subset_tokens"] == 3 * 32
    assert set(final["buckets"]) == {"0-512", "512-1536", "1536-4096"}
    assert len(final["decay"]["max"]) == 3
    assert json.loads((run / "provenance.json").read_text()) == {"piece_len": 33}
    records = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records if r["event"] == "eval"] == [2, 4]
    assert [r["event"] for r in records].count("final_eval") == 1
    # A rerun finds the run complete and does nothing more.
    assert main(argv) == 0
    rerun = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rerun].count("final_eval") == 1


def test_final_eval_records_what_produced_the_run(tmp_path):
    # The verdict checks each run against its role (deluge/eval/m1.py): the
    # run name is a hand-written label, so the result carries the config.
    argv = write_run_inputs(tmp_path, provenance={
        "dataset": "fake/ds", "revision": "r" * 40, "tokens": {"train": 40 * 33}})
    assert main(argv) == 0
    final = json.loads((tmp_path / "run" / "final_eval.json").read_text())
    assert {k: final[k] for k in ("seed", "total_tokens", "seq_len", "mixer", "data")} == {
        "seed": 5, "total_tokens": 256 * 4, "seq_len": 32, "mixer": "cfc",
        "data": {"dataset": "fake/ds", "revision": "r" * 40, "train_tokens": 40 * 33}}


def test_the_run_identity_names_the_mixer_type_and_no_data_without_provenance():
    from deluge.train.trainer import run_identity

    cfg = TrainConfig(**BUDGET)
    for path, kind in (("A0_scan_only", "cfc"), ("A0_transformer_baseline", "attention"),
                       ("A1_conv_baseline", "gated_conv"), ("A1_lru_baseline", "lru")):
        identity = run_identity(cfg, load_model_config(REPO / f"configs/ablations/{path}.yaml"),
                                None)
        assert identity["mixer"] == kind and identity["data"] is None


def test_a_resume_onto_other_data_is_refused(tmp_path):
    from deluge.train.loop import ResumeMismatch

    argv = write_run_inputs(tmp_path, provenance={
        "dataset": "fake/ds", "revision": "r" * 40, "tokens": {"train": 40 * 33}})
    assert main(argv) == 0
    provenance = tmp_path / "data" / "provenance.json"
    provenance.write_text(json.dumps({**json.loads(provenance.read_text()),
                                      "revision": "s" * 40}))
    with pytest.raises(ResumeMismatch, match="data_fingerprint"):
        main(argv)


def test_the_data_fingerprint_is_the_provenance_or_the_file_size(tmp_path):
    from deluge.train.trainer import data_fingerprint

    data = tmp_path / "train.bin"
    data.write_bytes(b"\0" * 66)
    assert json.loads(data_fingerprint(data, None)) == {"bytes": 66}
    assert json.loads(data_fingerprint(data, {
        "dataset": "d", "revision": "r", "piece_len": 33, "tokens": {"train": 7},
        "files": "ignored"})) == {"dataset": "d", "revision": "r", "piece_len": 33,
                                  "train_tokens": 7}


def test_data_tokenized_past_the_models_vocab_is_refused(tmp_path):
    # TINY_MODEL's vocab is 128: an id of 200 would index past the embedding.
    argv = write_run_inputs(tmp_path, provenance={"vocab_size": 200})
    with pytest.raises(ValueError, match="vocab_size 200.*vocab_size 128"):
        main(argv)
    assert not (tmp_path / "run" / "log.jsonl").exists()      # refused before it started


def test_a_heldout_set_without_a_piece_length_is_refused(tmp_path):
    argv = write_run_inputs(tmp_path)
    (tmp_path / "data" / "provenance.json").write_text("{}")
    with pytest.raises(ValueError, match="piece_len"):
        main(argv)


def test_pieces_that_do_not_fit_the_sequence_length_stop_the_run(tmp_path):
    with pytest.raises(ValueError, match="pieces of 65"):
        main(write_run_inputs(tmp_path, piece_len=65))


# --------------------------------------------------------------------------- #
# A real run's output through the exporter into the verdict
# --------------------------------------------------------------------------- #

def test_a_trained_run_exports_into_a_verdict_the_rules_can_judge(tmp_path):
    # The keys crossing trainer -> exporter -> verdict (full_loss, subset_loss,
    # step, skipped_steps, decay.min/max, the eval records' subset_loss) are
    # otherwise held together only by hand-written fixtures. A renamed decay
    # key would silently drop design 7.3's decay criterion: hence decay_max.
    from test_chain_export import DONE, FakeKaggle

    from deluge.chain.export import export_results
    from deluge.eval.m1 import load_run, stability

    # The tokens are random, so there is nothing to learn and the held-out loss
    # only drifts: at lr 1e-2 it drifts past the 0.02 late-divergence
    # tolerance in four steps, at 1e-4 by ~4e-5.
    argv = write_run_inputs(tmp_path, {"lr": 1e-4})
    assert main(argv) == 0
    run = tmp_path / "run"
    kernel = FakeKaggle({name: (run / name).read_bytes()
                         for name in ("final_eval.json", "log.jsonl", "provenance.json")})
    loaded = load_run(export_results(DONE, kernel, tmp_path / "results"))
    assert stability(loaded) == []
    assert loaded.decay_max and loaded.decay_min
    assert loaded.best_in_run_subset is not None and loaded.steps == 4
    assert loaded.mixer == "cfc" and loaded.seq_len == 32
