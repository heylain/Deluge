"""Export finished runs: results into the repo, checkpoints to Hugging Face.

Run hourly by .github/workflows/export.yml, apart from the chain's tick so a
tick never waits on a multi-GB transfer (M1 design 9). Idempotent, with no
ledger of its own: the work is read off the done records in
.github/chain/state.json, and each of a run's two steps is skipped once its
marker is in the repo.

  results     docs/results/runs/<name>/: final_eval.json (the marker),
              evals.csv and a downsampled train.csv, from the output of the
              kernel that ran the last session. The workflow commits them at
              once, so a verdict never waits on Hugging Face.
  checkpoint  the newest step-*.pt, the full log.jsonl and a run.json to
              <name>/ in the private HF repo configs/chain/export.yaml names;
              then docs/results/runs/<name>/export.json (the marker).

A failed step writes nothing and the next hourly run tries again. The exit
code is 1 -- the failed workflow emails you -- only for a run done more than
GRACE ago whose step still fails, so a transient Kaggle or HF error retries
quietly.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from .kaggle import KaggleCLI
from .step import kernel_slug

STATE = Path(".github/chain/state.json")
QUEUE = Path("configs/chain/runs.yaml")
CONFIG = Path("configs/chain/export.yaml")
RESULTS = Path("docs/results/runs")
GRACE = timedelta(hours=3)
STEPS = ("results", "checkpoint")
MARKERS = {"results": "final_eval.json", "checkpoint": "export.json"}
TRAIN_ROWS = 500
TRAIN_COLUMNS = ("step", "tokens_seen", "loss", "lr", "loss_scale", "skipped_steps",
                 "grad_norm", "tokens_per_second", "peak_alloc_gib", "peak_reserved_gib")
RESULTS_FILES = r"(^|/)(final_eval\.json|log\.jsonl)$"
CHECKPOINT_FILES = r"(^|/)(step-\d{9}\.pt|log\.jsonl|provenance\.json)$"


class ExportError(RuntimeError):
    """A step could not finish. It wrote nothing; the next run retries it."""


# ---- what to do ------------------------------------------------------------ #

def due(state: Dict[str, Any], step: str, root: Path) -> List[Dict[str, Any]]:
    """Done records that asked for export and still lack this step's marker."""
    return [record for record in state["done"]
            if record.get("export") and not record.get("skipped") and record.get("side")
            and not (root / record["name"] / MARKERS[step]).exists()]


def overdue(record: Dict[str, Any], now: datetime) -> bool:
    return now - datetime.fromisoformat(record["finished"]) > GRACE


# ---- log.jsonl to csv -------------------------------------------------------- #

def read_log(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(step records, eval records), each deduplicated by step -- a resumed
    session replays steps since its checkpoint, and the last write wins --
    and sorted by step."""
    steps: Dict[int, Dict[str, Any]] = {}
    evals: Dict[int, Dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue                     # a line cut off by a kill
        if record.get("event") == "step":
            steps[record["step"]] = record
        elif record.get("event") == "eval":
            evals[record["step"]] = record
    return [steps[k] for k in sorted(steps)], [evals[k] for k in sorted(evals)]


def downsample(rows: List[Dict[str, Any]], limit: int = TRAIN_ROWS) -> List[Dict[str, Any]]:
    """Every k-th row, so at most about `limit`, always keeping the last."""
    if len(rows) <= limit:
        return rows
    every = math.ceil(len(rows) / limit)
    kept = rows[::every]
    return kept if kept[-1] is rows[-1] else kept + [rows[-1]]


def eval_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """One evals.csv row: the subset loss, the buckets, and per recurrent layer
    i the decay summary (a_min_i, a_median_i, a_max_i) and h_rms_i."""
    row = {"step": record["step"], "tokens_seen": record.get("tokens_seen"),
           "subset_loss": record["subset_loss"]}
    for bucket, loss in (record.get("buckets") or {}).items():
        row[f"loss_{bucket.replace('-', '_')}"] = loss
    decay = record.get("decay") or {}
    for i, (lo, mid, hi) in enumerate(zip(decay.get("min", []), decay.get("median", []),
                                          decay.get("max", []))):
        row.update({f"a_min_{i}": lo, f"a_median_{i}": mid, f"a_max_{i}": hi})
    for i, rms in enumerate(record.get("state_rms") or []):
        row[f"h_rms_{i}"] = rms
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]], columns=None) -> None:
    columns = list(columns or (rows[0].keys() if rows else []))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---- the two steps ------------------------------------------------------------ #

def _find(directory: Path, name: str) -> Path:
    found = sorted(directory.rglob(name))
    if not found:
        raise ExportError(f"the kernel output has no {name}")
    return found[0]


def export_results(record: Dict[str, Any], kaggle, root: Path) -> Path:
    slug = kernel_slug(record["name"], record["side"])
    with tempfile.TemporaryDirectory() as tmp:
        kaggle.fetch_output(slug, Path(tmp), RESULTS_FILES)
        final = _find(Path(tmp), "final_eval.json")
        steps, evals = read_log(_find(Path(tmp), "log.jsonl"))
        out = root / record["name"]
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "train.csv", downsample(steps), TRAIN_COLUMNS)
        write_csv(out / "evals.csv", [eval_row(r) for r in evals])
        shutil.copy2(final, out / "final_eval.json")        # last: the marker
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hf_upload(folder: Path, repo: str, name: str) -> None:
    """Upload folder to <repo>/<name>/ in one commit, creating repo private."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ExportError("HF_TOKEN is not set (a repo secret, in the workflow)")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(folder), path_in_repo=name, repo_id=repo,
                      repo_type="model", commit_message=f"{name}: checkpoint, log, run.json")


def export_checkpoint(record: Dict[str, Any], kaggle, upload: Callable, root: Path,
                      hf_repo: Optional[str], entry: Optional[Dict[str, Any]],
                      owner: str) -> Path:
    if not hf_repo:
        raise ExportError(f"set hf_repo in {CONFIG}: the checkpoint has nowhere to go")
    name, slug = record["name"], kernel_slug(record["name"], record["side"])
    with tempfile.TemporaryDirectory() as tmp:
        fetched = Path(tmp) / "fetched"
        kaggle.fetch_output(slug, fetched, CHECKPOINT_FILES)
        checkpoints = sorted(fetched.rglob("step-*.pt"), key=lambda p: p.name)
        if not checkpoints:
            raise ExportError(f"the output of {slug} has no step-*.pt")
        newest = checkpoints[-1]
        stage = Path(tmp) / "stage"
        stage.mkdir()
        shutil.move(str(newest), stage / newest.name)
        shutil.move(str(_find(fetched, "log.jsonl")), stage / "log.jsonl")
        provenance = sorted(fetched.rglob("provenance.json"))
        (stage / "run.json").write_text(json.dumps({
            "run": entry, "done": record,
            "provenance": json.loads(provenance[0].read_text()) if provenance else None,
        }, indent=2) + "\n")
        size, digest = (stage / newest.name).stat().st_size, sha256(stage / newest.name)
        upload(stage, hf_repo, name)
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "export.json").write_text(json.dumps({
        "hf_repo": hf_repo, "path": f"{name}/{newest.name}", "sha256": digest,
        "size": size, "source_kernel": f"{owner}/{slug}"}, indent=2) + "\n")
    return out


# ---- one run of the workflow step ------------------------------------------- #

def queue_entry(queue: Path, name: str) -> Optional[Dict[str, Any]]:
    """The run's entry in runs.yaml, unvalidated: a typo in a later entry must
    not block exporting an earlier run."""
    runs = (yaml.safe_load(queue.read_text()) or {}).get("runs") or []
    return next((run for run in runs if isinstance(run, dict) and run.get("name") == name), None)


def run_step(step: str, state: Dict[str, Any], kaggle, root: Path, now: datetime,
             upload: Callable = hf_upload, hf_repo: Optional[str] = None,
             queue: Path = QUEUE, owner: str = "") -> int:
    code = 0
    for record in due(state, step, root):
        name = record["name"]
        try:
            if step == "results":
                export_results(record, kaggle, root)
            else:
                export_checkpoint(record, kaggle, upload, root, hf_repo,
                                  queue_entry(queue, name), owner)
        except Exception as error:  # noqa: BLE001 - any failure: retry next hour
            late = overdue(record, now)
            print(f"[export] {name}: {step} failed{' (overdue)' if late else ''}: "
                  f"{error!r}", file=sys.stderr)
            code = 1 if late else code
            continue
        print(f"[export] {name}: {step} exported")
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--step", choices=STEPS, required=True)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--root", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    owner = os.environ.get("KAGGLE_USERNAME")
    if not owner:
        parser.error("KAGGLE_USERNAME is not set (a repo secret, in the workflow)")
    config = yaml.safe_load(args.config.read_text()) or {}
    state = json.loads(args.state.read_text())
    return run_step(args.step, state, KaggleCLI(owner), args.root,
                    datetime.now(timezone.utc), hf_repo=config.get("hf_repo"),
                    queue=args.queue, owner=owner)


if __name__ == "__main__":
    sys.exit(main())
