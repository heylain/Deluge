"""Checkpoint durability: the host will kill this process without warning."""

import numpy as np
import pytest

from deluge.train.checkpoint import Checkpointer, TMP_PREFIX


class Box:
    def __init__(self, value=0.0):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def test_round_trips_state(tmp_path):
    ckpt = Checkpointer(tmp_path)
    ckpt.save(7, {"box": Box(3.5)}, {"note": "hello"})

    restored = Box()
    payload = ckpt.load_latest({"box": restored})
    assert restored.value == 3.5
    assert payload["step"] == 7
    assert payload["meta"]["note"] == "hello"


def test_leaves_no_temporary_files_behind(tmp_path):
    ckpt = Checkpointer(tmp_path)
    ckpt.save(1, {"box": Box(1.0)})
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(TMP_PREFIX)]


def test_rotation_keeps_only_the_newest(tmp_path):
    ckpt = Checkpointer(tmp_path, keep_last=2)
    for step in range(5):
        ckpt.save(step, {"box": Box(float(step))})
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "step-000000003.pt", "step-000000004.pt"]


def test_rotation_never_deletes_the_last_checkpoint(tmp_path):
    ckpt = Checkpointer(tmp_path, keep_last=1)
    for step in range(3):
        ckpt.save(step, {"box": Box(float(step))})
    assert len(list(tmp_path.iterdir())) == 1
    assert ckpt.load_latest({"box": Box()})["step"] == 2


def test_rejects_keep_last_below_one(tmp_path):
    with pytest.raises(ValueError, match="keep_last"):
        Checkpointer(tmp_path, keep_last=0)


def test_a_corrupt_newest_checkpoint_falls_back_to_the_one_before(tmp_path):
    # 11 h of training is worth more than the last few minutes of it.
    ckpt = Checkpointer(tmp_path, keep_last=3)
    ckpt.save(1, {"box": Box(1.0)})
    ckpt.save(2, {"box": Box(2.0)})
    (tmp_path / "step-000000002.pt").write_bytes(b"truncated garbage")

    restored = Box()
    payload = ckpt.load_latest({"box": restored})
    assert payload["step"] == 1
    assert restored.value == 1.0


def test_a_checkpoint_missing_expected_state_is_skipped(tmp_path):
    ckpt = Checkpointer(tmp_path, keep_last=3)
    ckpt.save(1, {"box": Box(1.0), "opt": Box(9.0)})
    ckpt.save(2, {"box": Box(2.0)})          # written without the optimizer

    box, opt = Box(), Box()
    payload = ckpt.load_latest({"box": box, "opt": opt})
    assert payload["step"] == 1
    assert (box.value, opt.value) == (1.0, 9.0)


def test_returns_none_when_there_is_nothing_to_resume(tmp_path):
    assert Checkpointer(tmp_path).load_latest({"box": Box()}) is None


def test_resumes_from_a_read_only_seed_directory(tmp_path):
    # The Kaggle shape: the previous session's output is mounted as input.
    seed_dir = tmp_path / "input"
    Checkpointer(seed_dir).save(4, {"box": Box(4.0)})

    out = tmp_path / "working"
    restored = Box()
    payload = Checkpointer(out).load_latest({"box": restored}, search_dirs=[seed_dir])
    assert payload["step"] == 4 and restored.value == 4.0


def test_the_sessions_own_output_wins_over_the_seed_snapshot(tmp_path):
    seed_dir = tmp_path / "input"
    out = tmp_path / "working"
    Checkpointer(seed_dir).save(4, {"box": Box(40.0)})
    ckpt = Checkpointer(out)
    ckpt.save(4, {"box": Box(4.0)})

    restored = Box()
    ckpt.load_latest({"box": restored}, search_dirs=[seed_dir])
    assert restored.value == 4.0


def test_rng_state_is_restored(tmp_path):
    ckpt = Checkpointer(tmp_path)
    np.random.seed(0)
    ckpt.save(1, {"box": Box()})
    expected = np.random.randn(4)

    np.random.seed(12345)                    # drift the generator away
    ckpt.load_latest({"box": Box()})
    assert np.array_equal(np.random.randn(4), expected)
def test_newest_readable_skips_a_corrupt_newest(tmp_path):
    ckpt = Checkpointer(tmp_path, keep_last=3)
    ckpt.save(1, {"box": Box(1.0)})
    newest = ckpt.save(2, {"box": Box(2.0)})
    newest.write_bytes(b"not a checkpoint")
    step, path = ckpt.newest_readable()
    assert step == 1 and path.name == "step-000000001.pt"


def test_newest_readable_searches_seed_dirs(tmp_path):
    seed = Checkpointer(tmp_path / "seed")
    seed.save(5, {"box": Box(5.0)})
    step, _ = Checkpointer(tmp_path / "out").newest_readable([tmp_path / "seed"])
    assert step == 5


def test_newest_readable_is_none_when_nothing_loads(tmp_path):
    assert Checkpointer(tmp_path).newest_readable() is None
