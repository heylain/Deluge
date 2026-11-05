"""The exporter (M1 design 9) against a fake Kaggle and a fake Hugging Face."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from deluge.chain.export import (GRACE, TRAIN_ROWS, downsample, due, eval_row,
                                 export_checkpoint, export_results, queue_entry,
                                 read_log, run_step)

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
DONE = {"name": "m1-a0-cfc-s1", "run_id": "m1-a0-cfc-s1@t", "commit": "abc123",
        "sessions": 1, "finished": (NOW - timedelta(hours=1)).isoformat(timespec="seconds"),
        "side": "a", "export": True}
FINAL = {"step": 1907, "full_loss": 3.21, "subset_loss": 3.2, "skipped_steps": 2}


def log_lines(n_steps=1200):
    lines = [json.dumps({"event": "start", "step": 0})]
    for step in range(1, n_steps + 1):
        lines.append(json.dumps({"event": "step", "step": step, "loss": 10.0 / step,
                                 "lr": 1e-4, "tokens_seen": step * 262144,
                                 "loss_scale": 65536.0, "skipped_steps": 0,
                                 "grad_norm": 0.5, "tokens_per_second": 20000.0,
                                 "peak_alloc_gib": 9.0, "peak_reserved_gib": 11.0}))
    # A resumed session replays steps 1001-1002 with different losses.
    lines += [json.dumps({"event": "step", "step": s, "loss": 1.0, "lr": 1e-4})
              for s in (1001, 1002)]
    for step in (190, 380):
        lines.append(json.dumps({"event": "eval", "step": step, "tokens_seen": step * 262144,
                                 "subset_loss": 4.0, "buckets": {"0-512": 4.1,
                                 "512-1536": 4.0, "1536-4096": 3.9},
                                 "decay": {"min": [0.9, 0.8], "median": [0.95, 0.9],
                                           "max": [0.999, 0.99]},
                                 "state_rms": [1.0, 1.1]}))
    lines.append(json.dumps({"event": "eval", "step": 380, "subset_loss": 3.9}))
    lines.append('{"event": "step", "step": 1201, "lo')          # cut off by a kill
    return "\n".join(lines) + "\n"


class FakeKaggle:
    """Writes a kernel output under dest the way kaggle 2.2.4 does: at the
    file's path inside the output (runs/<name>/...), or flat if nested=False."""

    def __init__(self, files, nested=True, error=None):
        self.files, self.nested, self.error = files, nested, error
        self.calls = []

    def fetch_output(self, slug, dest, pattern):
        import re
        self.calls.append((slug, pattern))
        if self.error:
            raise self.error
        for name, content in self.files.items():
            path = f"runs/{DONE['name']}/{name}" if self.nested else name
            if re.search(pattern, path):
                target = Path(dest) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content if isinstance(content, bytes) else content.encode())


def kernel_files():
    return {"final_eval.json": json.dumps(FINAL), "log.jsonl": log_lines(),
            "provenance.json": json.dumps({"piece_len": 4097}),
            "step-000001905.pt": b"older", "step-000001907.pt": b"newest",
            "session.json": "{}"}


def state(*records):
    return {"done": list(records)}


# ---- selection ----------------------------------------------------------------- #

def test_only_exportable_records_without_a_marker_are_due(tmp_path):
    records = [DONE,
               {**DONE, "name": "smoke", "export": False},
               {**DONE, "name": "old"} | {"side": None},
               {**DONE, "name": "skipped", "skipped": True},
               {**DONE, "name": "legacy"} | {"export": None}]
    assert [r["name"] for r in due(state(*records), "results", tmp_path)] == ["m1-a0-cfc-s1"]
    (tmp_path / "m1-a0-cfc-s1").mkdir()
    (tmp_path / "m1-a0-cfc-s1" / "final_eval.json").write_text("{}")
    assert due(state(DONE), "results", tmp_path) == []
    # Each step has its own marker: the checkpoint is still due.
    assert due(state(DONE), "checkpoint", tmp_path) == [DONE]


# ---- results ------------------------------------------------------------------- #

def test_the_log_is_deduplicated_last_write_wins(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(log_lines())
    steps, evals = read_log(path)
    assert [s["step"] for s in steps] == list(range(1, 1201))
    assert steps[1000]["loss"] == 1.0                      # step 1001, replayed
    assert [e["step"] for e in evals] == [190, 380] and evals[1]["subset_loss"] == 3.9


def test_downsampling_keeps_about_the_limit_and_the_last_row():
    rows = [{"step": i} for i in range(1, 1908)]
    kept = downsample(rows)
    assert len(kept) <= TRAIN_ROWS + 1 and kept[0]["step"] == 1 and kept[-1]["step"] == 1907
    assert downsample(rows[:10]) == rows[:10]


def test_an_eval_row_flattens_buckets_and_per_layer_monitors():
    row = eval_row({"step": 190, "tokens_seen": 5, "subset_loss": 4.0,
                    "buckets": {"0-512": 4.1}, "decay": {"min": [0.9], "median": [0.95],
                                                         "max": [0.99]}, "state_rms": [1.2]})
    assert row == {"step": 190, "tokens_seen": 5, "subset_loss": 4.0, "loss_0_512": 4.1,
                   "a_min_0": 0.9, "a_median_0": 0.95, "a_max_0": 0.99, "h_rms_0": 1.2}


@pytest.mark.parametrize("nested", [True, False])
def test_results_are_written_from_the_last_sessions_kernel(tmp_path, nested):
    # Review focus: whether the CLI reports runs/<name>/x or x, the files are found.
    kaggle = FakeKaggle(kernel_files(), nested=nested)
    out = export_results(DONE, kaggle, tmp_path)
    assert kaggle.calls[0][0] == "deluge-m1-a0-cfc-s1-a"
    assert json.loads((out / "final_eval.json").read_text()) == FINAL
    train = (out / "train.csv").read_text().splitlines()
    assert train[0].startswith("step,tokens_seen,loss,lr,loss_scale,skipped_steps")
    assert len(train) - 1 <= TRAIN_ROWS + 1
    evals = (out / "evals.csv").read_text().splitlines()
    assert len(evals) == 3 and "a_max_1" in evals[0]


def test_results_without_a_final_eval_write_nothing(tmp_path):
    files = kernel_files()
    del files["final_eval.json"]
    with pytest.raises(Exception, match="final_eval.json"):
        export_results(DONE, FakeKaggle(files), tmp_path)
    assert not (tmp_path / DONE["name"] / "final_eval.json").exists()


def test_a_results_step_that_fails_on_a_row_writes_nothing(tmp_path):
    # Design 9: a failed step writes nothing. The workflow commits whatever is
    # under docs/results/runs, marker or not, so a train.csv written before an
    # eval row failed would publish half a result.
    files = kernel_files()
    files["log.jsonl"] += json.dumps({"event": "eval", "step": 570}) + "\n"   # no subset_loss
    with pytest.raises(KeyError, match="subset_loss"):
        export_results(DONE, FakeKaggle(files), tmp_path)
    assert not (tmp_path / DONE["name"]).exists()


def test_the_results_marker_is_moved_into_place_last(tmp_path, monkeypatch):
    import shutil

    moved, real = [], shutil.move

    def move(src, dst):
        moved.append(Path(dst).name)
        return real(src, dst)

    monkeypatch.setattr(shutil, "move", move)
    export_results(DONE, FakeKaggle(kernel_files()), tmp_path)
    assert moved[-1] == "final_eval.json" and sorted(moved) == [
        "evals.csv", "final_eval.json", "train.csv"]


# ---- checkpoint ---------------------------------------------------------------- #

def test_the_newest_checkpoint_goes_to_hf_with_its_log_and_run_json(tmp_path):
    uploads = []

    def upload(folder, repo, name):
        uploads.append((repo, name, sorted(p.name for p in folder.iterdir()),
                        json.loads((folder / "run.json").read_text()),
                        (folder / "step-000001907.pt").read_bytes()))

    entry = {"name": DONE["name"], "model": "configs/ablations/A0_scan_only.yaml"}
    out = export_checkpoint(DONE, FakeKaggle(kernel_files()), upload, tmp_path,
                            "heylain/deluge-runs", entry, "heylaine")
    [(repo, name, files, run_json, content)] = uploads
    assert (repo, name) == ("heylain/deluge-runs", "m1-a0-cfc-s1")
    assert files == ["log.jsonl", "run.json", "step-000001907.pt"] and content == b"newest"
    assert run_json["run"] == entry and run_json["done"] == DONE
    assert run_json["provenance"] == {"piece_len": 4097}
    marker = json.loads((out / "export.json").read_text())
    assert marker["path"] == "m1-a0-cfc-s1/step-000001907.pt" and marker["size"] == 6
    assert marker["source_kernel"] == "heylaine/deluge-m1-a0-cfc-s1-a"
    assert len(marker["sha256"]) == 64


def test_no_hf_repo_fails_the_checkpoint_step_only(tmp_path):
    with pytest.raises(Exception, match="hf_repo"):
        export_checkpoint(DONE, FakeKaggle(kernel_files()), None, tmp_path, None, None, "o")
    assert not (tmp_path / DONE["name"] / "export.json").exists()


def test_a_failed_upload_writes_no_marker(tmp_path):
    def upload(*_):
        raise RuntimeError("HF is down")

    with pytest.raises(RuntimeError):
        export_checkpoint(DONE, FakeKaggle(kernel_files()), upload, tmp_path,
                          "heylain/deluge-runs", None, "o")
    assert not (tmp_path / DONE["name"] / "export.json").exists()


# ---- one workflow run -------------------------------------------------------------- #

def test_a_transient_failure_is_quiet_until_the_grace_runs_out(tmp_path):
    # Review focus: an expired HF token or a Kaggle outage must not email
    # every hour from the first; it must once the run has waited past GRACE.
    broken = FakeKaggle({}, error=RuntimeError("Kaggle is down"))
    assert run_step("results", state(DONE), broken, tmp_path, NOW) == 0
    late = {**DONE, "finished": (NOW - GRACE - timedelta(minutes=1)).isoformat()}
    assert run_step("results", state(late), broken, tmp_path, NOW) == 1


def test_a_bug_in_the_exporter_prints_its_traceback(tmp_path, capsys):
    # Unattended, a programming error otherwise surfaces GRACE later as one
    # repr line in the workflow log.
    files = kernel_files()
    files["log.jsonl"] += json.dumps({"event": "eval", "step": 570}) + "\n"   # no subset_loss
    run_step("results", state(DONE), FakeKaggle(files), tmp_path, NOW)
    err = capsys.readouterr().err
    assert "Traceback" in err and "eval_row" in err


def test_an_export_error_is_reported_without_a_traceback(tmp_path, capsys):
    files = kernel_files()
    del files["final_eval.json"]
    run_step("results", state(DONE), FakeKaggle(files), tmp_path, NOW)
    err = capsys.readouterr().err
    assert "has no final_eval.json" in err and "Traceback" not in err


def test_a_step_already_done_is_not_repeated(tmp_path):
    kaggle = FakeKaggle(kernel_files())
    assert run_step("results", state(DONE), kaggle, tmp_path, NOW) == 0
    assert run_step("results", state(DONE), kaggle, tmp_path, NOW) == 0
    assert len(kaggle.calls) == 1


def test_results_do_not_wait_on_the_checkpoint(tmp_path):
    kaggle = FakeKaggle(kernel_files())
    late = {**DONE, "finished": (NOW - GRACE * 2).isoformat()}
    assert run_step("checkpoint", state(late), kaggle, tmp_path, NOW, hf_repo=None) == 1
    assert run_step("results", state(late), kaggle, tmp_path, NOW) == 0
    assert (tmp_path / DONE["name"] / "final_eval.json").exists()


def test_the_queue_entry_is_read_without_validating_other_entries():
    text = "runs:\n  - name: m1-a0-cfc-s1\n    model: m.yaml\n  - name: Bad_Name\n"
    assert queue_entry(text, "m1-a0-cfc-s1") == {"name": "m1-a0-cfc-s1", "model": "m.yaml"}
    assert queue_entry(text, "gone") is None


def queued(model):
    return f"runs:\n  - name: m1-a0-cfc-s1\n    model: {model}\n"


def uploaded_run_json(tmp_path, record, shown):
    """The run.json a checkpoint step uploads, with runs.yaml edited since the
    run started; shown records what was read from git."""
    queue = tmp_path / "runs.yaml"
    queue.write_text(queued("edited-since.yaml"))
    uploads = []

    def show(commit, path):
        shown.append((commit, path))
        return queued("as-it-ran.yaml")

    def upload(folder, repo, name):
        uploads.append(json.loads((folder / "run.json").read_text()))

    assert run_step("checkpoint", state(record), FakeKaggle(kernel_files()), tmp_path / "r",
                    NOW, upload=upload, hf_repo="heylain/deluge-runs", queue=queue,
                    owner="o", show=show) == 0
    [run_json] = uploads
    return run_json


def test_run_json_takes_the_queue_entry_at_the_runs_pinned_commit(tmp_path):
    # runs.yaml: "editing it afterwards only matters for runs not yet started",
    # so today's entry need not be what ran.
    shown = []
    run_json = uploaded_run_json(tmp_path, DONE, shown)
    assert run_json["run"] == {"name": "m1-a0-cfc-s1", "model": "as-it-ran.yaml"}
    assert shown == [("abc123", "configs/chain/runs.yaml")]


def test_a_record_with_no_commit_falls_back_to_todays_queue(tmp_path):
    shown = []
    record = {k: v for k, v in DONE.items() if k != "commit"}
    run_json = uploaded_run_json(tmp_path, record, shown)
    assert run_json["run"] == {"name": "m1-a0-cfc-s1", "model": "edited-since.yaml"}
    assert shown == []
