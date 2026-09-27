"""session.py: what one Kaggle kernel does. Run here with fake commands."""

import json
import subprocess
from pathlib import Path

import pytest

from deluge.chain import session, step
from deluge.train.checkpoint import Checkpointer


class Box:
    def __init__(self, value=0.0):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def params(accelerator="cpu", data="synthetic"):
    return {"run": {"name": "smoke", "model": "configs/model/screen.yaml",
                    "train": "configs/train/smoke.yaml",
                    "model_impl": "deluge.train.smoke:build",
                    "accelerator": accelerator, "data": data,
                    "commit": "abc123", "run_id": "smoke@t0"},
            "session": 2, "side": "b", "repo": "https://example.invalid/Deluge"}


class FakeCommands:
    """Stands in for subprocess.run: git and pip succeed, train does `train`."""

    def __init__(self, train=None, fail_on=None):
        self.calls, self.train, self.fail_on = [], train, fail_on

    def __call__(self, cmd, check=False, **kwargs):
        self.calls.append(cmd)
        if self.fail_on and self.fail_on in cmd:
            if check:
                raise subprocess.CalledProcessError(128, cmd)
            return subprocess.CompletedProcess(cmd, 128)
        if "deluge.train" in cmd:
            return subprocess.CompletedProcess(cmd, self.train(cmd) if self.train else 0)
        return subprocess.CompletedProcess(cmd, 0)


def session_json(work):
    return json.loads((work / "session.json").read_text())


def test_no_usable_gpu_reports_3_and_does_nothing_else(tmp_path):
    commands = FakeCommands()
    assert session.main(params("gpu"), work=tmp_path, input_root=tmp_path / "in",
                        src=tmp_path / "src", run_cmd=commands, gpus=lambda: 0) == 0
    assert session_json(tmp_path)["exit_code"] == step.NO_GPU
    assert commands.calls == []


def test_no_gpu_code_matches_the_orchestrator():
    assert session.NO_GPU == step.NO_GPU


def test_session_json_is_written_when_the_clone_fails(tmp_path):
    session.main(params(), work=tmp_path, input_root=tmp_path / "in",
                 src=tmp_path / "src", run_cmd=FakeCommands(fail_on="clone"),
                 gpus=lambda: 0)
    record = session_json(tmp_path)
    assert record["exit_code"] == 1
    assert record["run_id"] == "smoke@t0" and record["session"] == 2


def test_a_full_session_carries_forward_trains_and_reports(tmp_path):
    prior = tmp_path / "in" / "someuser" / "deluge-smoke-a" / "runs" / "smoke"
    Checkpointer(prior).save(40, {"box": Box(4.0)})
    (prior / "log.jsonl").write_text(json.dumps({"event": "step", "tokens_seen": 100}) + "\n")
    work = tmp_path / "work"

    def train(cmd):
        out = Path(cmd[cmd.index("--out") + 1])
        assert (out / "step-000000040.pt").exists(), "checkpoint carried before training"
        assert Path(cmd[cmd.index("--data") + 1]).stat().st_size > 0
        with open(out / "log.jsonl", "a") as log:
            log.write(json.dumps({"event": "step", "tokens_seen": 250}) + "\n")
            log.write(json.dumps({"event": "stop", "reason": "deadline"}) + "\n")
        return 2

    commands = FakeCommands(train=train)
    session.main({**params(), "first": False}, work=work, input_root=tmp_path / "in",
                 src=tmp_path / "src",
                 run_cmd=commands, gpus=lambda: 0)
    record = session_json(work)
    assert record["exit_code"] == 2 and record["tokens_seen"] == 250
    train_cmd = next(c for c in commands.calls if "deluge.train" in c)
    assert train_cmd[train_cmd.index("--device") + 1] == "cpu"
    assert "--data-parallel" not in train_cmd


def test_carry_forward_skips_a_corrupt_newest_checkpoint(tmp_path):
    prior = tmp_path / "prior"
    ckpt = Checkpointer(prior, keep_last=3)
    ckpt.save(1, {"box": Box(1.0)})
    ckpt.save(2, {"box": Box(2.0)}).write_bytes(b"garbage")
    carried = session.carry_forward(prior, tmp_path / "out")
    assert carried == tmp_path / "out" / "step-000000001.pt"


def test_carry_forward_with_no_prior_run_just_makes_the_dir(tmp_path):
    assert session.carry_forward(None, tmp_path / "out") is None
    assert (tmp_path / "out").is_dir()


def test_prior_run_dir_is_found_wherever_kaggle_mounts_it(tmp_path):
    (tmp_path / "a" / "b" / "runs" / "smoke").mkdir(parents=True)
    (tmp_path / "c" / "runs" / "other").mkdir(parents=True)
    assert session.find_prior_run_dir(tmp_path, "smoke") == tmp_path / "a/b/runs/smoke"
    # Kaggle has mounted a source both as input/<slug>/ and as
    # input/notebooks/<owner>/<slug>/ (spike A2); the glob must reach either.
    (tmp_path / "flat" / "deluge-x-a" / "runs" / "x").mkdir(parents=True)
    (tmp_path / "notebooks" / "me" / "deluge-y-a" / "runs" / "y").mkdir(parents=True)
    assert session.find_prior_run_dir(tmp_path, "x") == tmp_path / "flat/deluge-x-a/runs/x"
    assert session.find_prior_run_dir(tmp_path, "y") == tmp_path / "notebooks/me/deluge-y-a/runs/y"
    assert session.find_prior_run_dir(tmp_path, "missing") is None
    assert session.find_prior_run_dir(tmp_path / "nope", "smoke") is None


def test_tokens_seen_takes_the_last_step_record_and_tolerates_junk(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"tokens_seen": 5}\nnot json\n{"tokens_seen": 9}\n{"event": "stop"}\n')
    assert session.tokens_seen(log) == 9
    assert session.tokens_seen(tmp_path / "absent.jsonl") == 0


@pytest.mark.parametrize("accelerator, gpus, device, parallel", [
    ("cpu", 0, "cpu", False),
    ("gpu", 1, None, False),
    ("gpu", 2, None, True),
])
def test_train_command_fits_the_hardware(tmp_path, accelerator, gpus, device, parallel):
    run = params(accelerator)["run"]
    cmd = session.train_command(run, tmp_path / "t.bin", tmp_path / "out", gpus, tmp_path)
    assert ("--data-parallel" in cmd) is parallel
    assert (cmd[cmd.index("--device") + 1] if "--device" in cmd else None) == device
    assert cmd[cmd.index("--model-impl") + 1] == "deluge.train.smoke:build"


def test_dataset_file_is_found_under_input(tmp_path):
    target = tmp_path / "in" / "datasets" / "heylain" / "deluge-tokens" / "tokens.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\0\0")
    found = session.resolve_data("heylain/deluge-tokens:tokens.bin", tmp_path / "in", tmp_path)
    assert found == target
    with pytest.raises(FileNotFoundError, match="tokens.bin"):
        session.resolve_data("heylain/other:tokens.bin", tmp_path / "in", tmp_path)


def test_a_handed_off_session_with_no_prior_output_refuses_to_train(tmp_path):
    # Kaggle drops an unresolvable kernel source with only a warning (spike A7),
    # so "no prior output" on a handed-off session means the mount failed --
    # training from step 0 would silently throw the run away.
    commands = FakeCommands()
    handed_off = {**params(), "first": False}
    session.main(handed_off, work=tmp_path / "work", input_root=tmp_path / "in",
                 src=tmp_path / "src", run_cmd=commands, gpus=lambda: 0)
    assert session_json(tmp_path / "work")["exit_code"] == 1
    assert not any("deluge.train" in c for c in commands.calls)
