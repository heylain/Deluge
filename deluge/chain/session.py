"""One Kaggle session of a chained run. Pushed as the kernel's code file.

deluge.chain.kaggle.render writes this session's parameters into RUN below and
pushes the file; Kaggle runs it top to bottom. Until step 2 installs the
package it may use only the standard library and torch (which Kaggle images
carry), so nothing here imports deluge at module level.

  1. no usable GPU on a gpu run -> report 3 and stop (quota, or a P100)
  2. clone the pinned commit, pip install -e
  3. copy the other side's newest readable checkpoint and log into this output
     (and refuse to train if a handed-off session finds no such output)
  4. python -m deluge.train
  5. always: write /kaggle/working/session.json, exit 0 -- with code 4 if the
     session failed before step 3 finished, so the chain retries this side
     instead of handing off an output that holds no checkpoint

Exit 0 whatever happened, because Kaggle keeps a failed kernel's output
unreliably and the orchestrator reads the real exit code from session.json.
"""

import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

RUN = {}  # rendered by deluge.chain.kaggle.render; empty only in tests
RUN_MARKER = "RUN = {}"

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SRC = Path("/tmp/deluge-src")       # not under WORK: the clone must not bloat the output
MIN_CAPABILITY = (7, 0)             # Triton's floor; the P100 is (6, 0)
NO_GPU = 3                          # deluge.chain.step.NO_GPU; a test keeps them equal
NO_HANDOFF = 4                      # deluge.chain.step.NO_HANDOFF; likewise
SYNTHETIC_TOKENS = 200_000
SYNTHETIC_VOCAB = 32_000            # configs/base.yaml vocab_size


def usable_gpus() -> int:
    """CUDA devices this session can train on: compute capability 7.0+."""
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return sum(1 for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_capability(i) >= MIN_CAPABILITY)


def fetch(repo: str, commit: str, src: Path, run_cmd: Callable) -> None:
    if src.exists():
        shutil.rmtree(src)
    run_cmd(["git", "clone", "--quiet", repo, str(src)], check=True)
    run_cmd(["git", "-C", str(src), "checkout", "--quiet", commit], check=True)
    run_cmd([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(src)], check=True)
    # The editable install's .pth is only read at interpreter start.
    sys.path.insert(0, str(src))


def find_prior_run_dir(input_root: Path, name: str) -> Optional[Path]:
    """The other side's runs/<name>, wherever Kaggle mounted its output."""
    if not input_root.is_dir():
        return None
    found = sorted(p for p in input_root.glob(f"**/runs/{name}") if p.is_dir())
    return found[0] if found else None


def carry_forward(prior: Optional[Path], out_dir: Path) -> Optional[Path]:
    """Copy the prior session's newest readable checkpoint, and its log, into out_dir.

    This is the chain's invariant: every session's output holds the run's
    newest checkpoint, even a session that dies before its first save.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if prior is None:
        return None
    from deluge.train.checkpoint import Checkpointer

    if (prior / "log.jsonl").exists():
        shutil.copy2(prior / "log.jsonl", out_dir / "log.jsonl")
    found = Checkpointer(out_dir).newest_readable([prior])
    if found is None:
        return None
    _, path = found
    target = out_dir / path.name
    if path != target:
        shutil.copy2(path, target)
    return target


def resolve_data(data: str, input_root: Path, scratch: Path) -> Path:
    """The token file: generated into scratch (never the output), or the dataset's."""
    if data == "synthetic":
        import numpy as np
        path = scratch / "deluge-synthetic.bin"
        np.random.default_rng(0).integers(
            0, SYNTHETIC_VOCAB, size=SYNTHETIC_TOKENS, dtype=np.uint16).tofile(path)
        return path
    dataset, filename = data.split(":", 1)
    slug = dataset.split("/", 1)[1]
    found = sorted(input_root.glob(f"**/{slug}/{filename}"))
    if not found:
        raise FileNotFoundError(
            f"{filename} from dataset {dataset} is not under {input_root}; is the "
            f"dataset attached (dataset_sources) and is the file name right?")
    return found[0]


def train_command(run: dict, data: Path, out_dir: Path, gpus: int, src: Path) -> list:
    cmd = [sys.executable, "-m", "deluge.train",
           "--model", str(src / run["model"]), "--train", str(src / run["train"]),
           "--data", str(data), "--out", str(out_dir),
           "--model-impl", run["model_impl"]]
    if run["accelerator"] == "cpu":
        cmd += ["--device", "cpu"]
    elif gpus >= 2:
        cmd.append("--data-parallel")
    return cmd


def tokens_seen(log: Path) -> int:
    seen = 0
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "tokens_seen" in record:
                seen = record["tokens_seen"]
    return seen


def main(params: Optional[dict] = None, work: Path = WORK, input_root: Path = INPUT,
         src: Path = SRC, run_cmd: Callable = subprocess.run,
         gpus: Callable[[], int] = usable_gpus) -> int:
    params = params or RUN
    run = params["run"]
    out_dir = work / "runs" / run["name"]
    record = {"run_id": run["run_id"], "session": params["session"],
              "side": params["side"], "commit": run["commit"],
              "exit_code": 1, "tokens_seen": 0}
    carried = False                 # until then, a failure must not hand off
    try:
        found = gpus()
        if run["accelerator"] == "gpu" and found == 0:
            print("[session] no CUDA device of capability >= 7.0; reporting 3")
            record["exit_code"] = NO_GPU
            return 0
        fetch(params["repo"], run["commit"], src, run_cmd)
        prior = find_prior_run_dir(input_root, run["name"])
        if prior is None and not params.get("first", True):
            # Kaggle drops an unresolvable kernel source with only a warning
            # (docs/kaggle-chain-spike.md, A7). A handed-off session that sees
            # no prior output must not quietly train the run again from step 0.
            raise FileNotFoundError(
                f"expected the other side's runs/{run['name']} under {input_root} "
                f"and found none; refusing to start the run from scratch")
        checkpoint = carry_forward(prior, out_dir)
        carried = True
        print(f"[session] {run['run_id']} session {params['session']} on "
              f"{params['side']}, resuming from {checkpoint}")
        data = resolve_data(run["data"], input_root, src.parent)
        result = run_cmd(train_command(run, data, out_dir, found, src), cwd=str(src))
        record["exit_code"] = result.returncode
    except Exception:  # noqa: BLE001 - anything at all must still reach session.json
        traceback.print_exc()
        if not carried:
            record["exit_code"] = NO_HANDOFF
    finally:
        record["tokens_seen"] = tokens_seen(out_dir / "log.jsonl")
        # Whether this output can be resumed from. A session that died before
        # its first save leaves nothing (Kaggle drops empty dirs), and the
        # next one must then start afresh rather than look for a source.
        record["checkpoint"] = any(out_dir.glob("step-*.pt"))
        work.mkdir(parents=True, exist_ok=True)
        (work / "session.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"[session] {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
