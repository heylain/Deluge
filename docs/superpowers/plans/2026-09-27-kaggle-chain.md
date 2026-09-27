# Kaggle Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A scheduled GitHub Action that keeps Deluge training on Kaggle's free tier: it works through a queue of runs, one ~11 h session at a time, with each session resuming from the one before, and no local machine involved.

**Architecture:** Each run gets two private Kaggle script kernels (`deluge-<run>-a` and `-b`) that take turns. Each lists the other in `kernel_sources`, so a session mounts the previous session's output, checkpoint included. A pure decision function (`deluge/chain/step.py`) turns the state plus Kaggle's report into one action. A thin CLI adapter (`kaggle.py`) and a tick runner (`orchestrate.py`) do the I/O, and a 30-minute cron workflow runs the tick and commits `.github/chain/state.json` when it changes.

**Tech Stack:** Python ≥3.10 (the Actions runner uses 3.12), PyYAML, the `kaggle` CLI (≥1.7, Actions only), GitHub Actions, pytest. torch is used only inside Kaggle sessions and in the `importorskip` tests.

**Spec:** `docs/superpowers/specs/2026-09-27-kaggle-chain-design.md`

## Global Constraints

- $0: Kaggle free tier and GitHub Actions on the public repo `heylain/Deluge`, default branch `master`.
- No hyperparameter lives in Python: budgets and model shapes are YAML under `configs/` (`docs/project-structure.md` principle 4).
- `deluge/chain/` must be importable without torch. Only `session.py` touches torch, and only at call time.
- Kernels are private (`is_private: true`), with `enable_internet: true`.
- Kernel slugs are `deluge-<name>-a|b`. Run names match `^[a-z0-9]+(-[a-z0-9]+)*$` and are at most 41 characters (Kaggle titles are at most 50).
- Exit codes: `0` run complete, `2` resumable (these two come from `deluge.train`), `3` no usable GPU (from `session.py`), anything else is a crash.
- Timing: tick every 30 min · 10 min grace after a push · stuck after 13 h · quota backoff 6 h · halt at 2 failures in a row without progress · fail the workflow at 6 API errors in a row.
- The workflow has **no** `pull_request`/`pull_request_target`/`push` trigger.
- Commit trailer on every commit: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Match the codebase's style: module docstrings that say *why*, comments on non-obvious lines, error messages that name the fix.

## Review Focus

1. **A status read soon after a push can be the *previous* version's.** A manual tick minutes after a push may see `complete` plus an old `session.json`. Expected: nothing happens, no duplicate push. Pinned by `test_no_decision_within_grace_after_push` (Task 3).
2. **A session that exits 2 but trained nothing** (deadline shorter than startup, data path wrong but tolerated). Expected: it counts toward the halt, instead of looping and eating quota. Pinned by `test_exit_2_without_progress_counts_as_a_failure` (Task 3).
3. **A typo in `runs.yaml`** (a config path, a bad name). Expected: it fails in CI and in the test suite, not 10 minutes into a Kaggle session. Pinned by `test_committed_queue_loads_and_its_configs_parse` (Task 2).
4. **`kaggle kernels push` exits 0 but prints an error.** Expected: it's treated as a failed push, not recorded as a live session. Pinned by `test_push_without_success_line_raises` (Task 5).
5. **Status casing and prefix differences between CLI versions** (`"complete"` vs `"KernelWorkerStatus.COMPLETE"`). Expected: both map to `complete`. Pinned by `test_status_parses_enum_and_plain_forms` (Task 5).

---

## File Structure

| Path | Responsibility |
|---|---|
| `deluge/chain/__init__.py` | package marker plus a one-paragraph map of the modules |
| `deluge/chain/queue.py` | load and validate `configs/chain/runs.yaml` |
| `deluge/chain/step.py` | the pure decision: `step`, `api_error`, `push_refused`, state shape, constants |
| `deluge/chain/session.py` | the script a Kaggle kernel runs (stdlib-only at import) |
| `deluge/chain/kaggle.py` | `kaggle` CLI adapter plus `render()` of the kernel directory |
| `deluge/chain/orchestrate.py` | one tick: observe → step → act → write state; the CLI entry |
| `deluge/train/checkpoint.py` | **modify**: add `Checkpointer.newest_readable` |
| `deluge/train/smoke.py` | test-only torch model for the rehearsal |
| `configs/chain/runs.yaml` | the queue |
| `configs/train/smoke.yaml` | the rehearsal budget (three sessions) |
| `.github/chain/state.json` | the chain's state (initial: idle) |
| `.github/workflows/chain.yml` | the cron tick |
| `docs/kaggle.md` | **modify**: add "The chain" section |
| `docs/project-structure.md` | **modify**: add `chain/` to the tree |
| `docs/kaggle-chain-spike.md` | spike findings (Task 1) |
| `tests/test_chain_queue.py`, `test_chain_step.py`, `test_chain_session.py`, `test_chain_kaggle.py`, `test_chain_orchestrate.py`, `test_chain_workflow.py` | tests |

---

### Task 0: Isolated worktree and baseline

The main checkout has the user's uncommitted M2 work in progress (`deluge/model/`, `deluge/kernels/`, modified configs). Don't touch it.

**Files:** none

- [ ] **Step 1: Create the worktree off the committed harness**

```bash
cd /home/LAIN/dev/Deluge
git worktree add ../Deluge-chain -b kaggle-chain m1-training-harness
```

- [ ] **Step 2: Confirm the worktree's package shadows the editable install**

```bash
cd /home/LAIN/dev/Deluge-chain
PYTHONPATH=$PWD ../Deluge/.venv/bin/python -c "import deluge; print(deluge.__file__)"
```

Expected: `/home/LAIN/dev/Deluge-chain/deluge/__init__.py`. If it prints the main checkout's path instead, make a worktree venv (`python -m venv .venv && .venv/bin/pip install -e '.[dev]' torch --index-url https://download.pytorch.org/whl/cpu`) and use `.venv/bin/python` wherever this plan says `$PY`.

From here on, `$PY` means `PYTHONPATH=$PWD ../Deluge/.venv/bin/python`, run from `/home/LAIN/dev/Deluge-chain`.

- [ ] **Step 3: Baseline**

Run: `$PY -m pytest -q`
Expected: all pass (189 or so; 1 skipped). `test_missing_model_module_points_at_m1` fails only in the main checkout, because there the untracked `deluge/model/` exists. In the worktree it must pass. Record the count.

---

### Task 1: Spike: confirm Kaggle's behaviour (throwaway)

**Needs the user:** a phone-verified Kaggle account and `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME`/`KAGGLE_KEY` exported). Ask before starting if neither exists. Everything here runs in the scratchpad, and nothing except the findings file is committed.

**Files:**
- Create: `docs/kaggle-chain-spike.md` (findings, committed)

The assumptions later tasks are built on are listed below. The last step compares each against what actually happened.

| # | Assumption | Used by |
|---|---|---|
| A1 | Two kernels may list each other in `kernel_sources` | design |
| A2 | A source kernel's output mounts somewhere under `/kaggle/input/`, and `session.py` finds `runs/<name>` with a `**` glob | Task 4 |
| A3 | The mount is the source's **latest** version | design |
| A4 | `kaggle kernels output <id> -p <dir> --file-pattern '^session\.json$'` downloads only that file | Task 5 |
| A5 | `kaggle kernels status <id>` prints `... has status "<X>"`, with X in queued/running/complete/error/cancel* (any case, optional `KernelWorkerStatus.` prefix) | Task 5 |
| A6 | A successful push prints a line containing `successfully pushed`, and a failed push either exits non-zero or omits that line | Task 5 |
| A7 | A push may reference a kernel that does not exist yet, or it fails. Either is fine: the design never does it, but record which | info |
| A8 | `machine_shape: NvidiaTeslaT4` + `enable_gpu: true` provisions T4(s), capability 7.5. Record whether it's one or two | Task 4 docs |

- [ ] **Step 1: Kaggle CLI in a scratch venv**

```bash
S=/tmp/claude-1000/-home-LAIN-dev-Deluge/fca2d9f7-f833-4298-a328-35dc1a962b0d/scratchpad/spike
mkdir -p $S && python -m venv $S/venv && $S/venv/bin/pip install -q "kaggle>=1.7"
K=$S/venv/bin/kaggle
$K --version && $K kernels list --mine --page-size 1
$K kernels output -h | grep -i pattern    # A4: does --file-pattern exist?
```

Expected: a version ≥1.7, and the list command succeeds (auth works).

- [ ] **Step 2: Write the spike kernel source**

`$S/spike.py`:

```python
import json, os, pathlib, subprocess
MARKER = "__MARKER__"
print("MARKER", MARKER)
print(subprocess.run(["find", "/kaggle/input", "-maxdepth", "7"], capture_output=True, text=True).stdout)
for m in pathlib.Path("/kaggle/input").glob("**/runs/spike/marker.txt"):
    print("SAW", m, m.read_text())
work = pathlib.Path("/kaggle/working")
(work / "runs" / "spike").mkdir(parents=True, exist_ok=True)
(work / "runs" / "spike" / "marker.txt").write_text(MARKER)
(work / "session.json").write_text(json.dumps({"marker": MARKER}))
(work / "big.bin").write_bytes(os.urandom(10_000_000))   # --file-pattern must skip this
if MARKER.startswith("err"):
    raise RuntimeError("deliberate")
if MARKER.startswith("gpu"):
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
    import torch
    print("CUDA", torch.cuda.device_count(),
          [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])
```

And a helper that renders a kernel dir and pushes it, `$S/push.sh`:

```bash
#!/usr/bin/env bash
# usage: push.sh <slug> <marker> <kernel_sources json array> [gpu]
set -u
S=$(dirname "$0"); K=$S/venv/bin/kaggle; OWNER=$($K config view | awk '/username/{print $3}')
D=$S/k-$1; rm -rf "$D"; mkdir -p "$D"
sed "s/__MARKER__/$2/" "$S/spike.py" > "$D/spike.py"
GPU=false; SHAPE=""; [ "${4:-}" = gpu ] && GPU=true && SHAPE=NvidiaTeslaT4
cat > "$D/kernel-metadata.json" <<EOF
{"id": "$OWNER/$1", "title": "$1", "code_file": "spike.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_internet": false,
 "enable_gpu": $GPU, "machine_shape": "$SHAPE",
 "dataset_sources": [], "competition_sources": [], "kernel_sources": $3}
EOF
$K kernels push -p "$D"; echo "PUSH EXIT $?"
```

If `kaggle config view` doesn't print the username, set `OWNER` by hand. Then `chmod +x $S/push.sh`.

- [ ] **Step 3: Run the sequence, recording raw output of every command into `$S/log.txt`**

```bash
cd $S; O=<owner>
./push.sh deluge-spike-a a1 '[]'                         # first push, no sources
# poll `$K kernels status $O/deluge-spike-a` every ~20 s until complete/error
# (Monitor tool or a loop, not `watch`); record every distinct string (A5)
mkdir o1 && $K kernels output $O/deluge-spike-a -p o1 --file-pattern '^session\.json$'; ls -la o1   # A4
./push.sh deluge-spike-b b1 "[\"$O/deluge-spike-a\"]"     # b sources a
# when b completes: fetch its log (kaggle kernels output ... -p ob, the .log file) -> mount path (A2), SAW a1
./push.sh deluge-spike-a a2 "[\"$O/deluge-spike-b\"]"     # a sources b: now mutual (A1)
./push.sh deluge-spike-b b2 "[\"$O/deluge-spike-a\"]"     # must SAW a2, not a1 (A3)
./push.sh deluge-spike-c err1 '[]'                       # errors after writing output
# record c's final status string; kaggle kernels output c: is session.json there? (info)
./push.sh deluge-spike-d d1 "[\"$O/deluge-spike-nope\"]"  # A7: record exit code + stdout
sed -i 's/"language": "python"/"language": "cobol"/' k-deluge-spike-d/kernel-metadata.json
$K kernels push -p k-deluge-spike-d; echo "PUSH EXIT $?"  # A6: how a failed push reports
./push.sh deluge-spike-g gpu1 '[]' gpu                   # A8: ~2 min of GPU quota
```

- [ ] **Step 4: Delete the spike kernels**

Try `$K kernels delete $O/<slug>` for each of a, b, c, d, g (it exists in recent CLIs; `-y` may be needed). If the CLI has no delete, list the slugs for the user to delete at kaggle.com → Your Work → Code.

- [ ] **Step 5: Write `docs/kaggle-chain-spike.md`**

One section per assumption A1–A8: verdict (holds / differs), the raw evidence line from `log.txt`, and the exact constant later tasks must use. The constants are: the mount glob, `STATUS_ALIASES`, the push success marker, `--file-pattern` availability, and the T4 count.

- [ ] **Step 6: Gate**

- **A1 or A3 fails:** the handoff design is wrong. **Stop and report to the user** with the evidence. Don't continue to Task 2.
- **A2, A4, A5 or A6 differs:** continue, but apply the finding in the owning task:
  - A2: the glob in `find_prior_run_dir` (Task 4).
  - A4: if `--file-pattern` doesn't exist, `fetch_session_json` downloads the whole output. Note it as a cost in the doc.
  - A5: `STATUS_ALIASES` (Task 5).
  - A6: `PUSH_OK` (Task 5).
- **A8 shows one T4:** tell the user. Nothing in the chain changes, but `docs/kaggle.md`'s session table doubles (Task 7).

- [ ] **Step 7: Commit the findings**

```bash
git add docs/kaggle-chain-spike.md
git commit -m "docs: Kaggle chain spike findings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Queue, smoke model and smoke budget

**Files:**
- Create: `deluge/chain/__init__.py`, `deluge/chain/queue.py`, `deluge/train/smoke.py`, `configs/chain/runs.yaml`, `configs/train/smoke.yaml`
- Test: `tests/test_chain_queue.py`

**Interfaces:**
- Produces: `load_queue(path) -> list[dict]`. Each dict has keys `name, model, train, model_impl, accelerator, data`, with `model_impl` defaulting to `"deluge.model:build"`. It raises `deluge.config.ConfigError`.
- Produces: `deluge.train.smoke.build(model_cfg) -> nn.Module` whose `forward(x, y)` returns a scalar loss; `build_crashing(model_cfg)` raises `RuntimeError`; `SECONDS_PER_STEP: float`.

- [ ] **Step 1: Write the failing tests**

`tests/test_chain_queue.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_chain_queue.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'deluge.chain'`.

- [ ] **Step 3: Implement**

`deluge/chain/__init__.py`:

```python
"""Unattended training across Kaggle's 12 h sessions.

queue.py reads configs/chain/runs.yaml; step.py decides what to do next and
does no I/O; kaggle.py talks to the kaggle CLI; orchestrate.py is one tick of
.github/workflows/chain.yml; session.py is what a Kaggle kernel runs. Design:
docs/superpowers/specs/2026-09-27-kaggle-chain-design.md.

Importable without torch: the orchestrator runs on a GitHub runner with the
base install only.
"""
```

`deluge/chain/queue.py`:

```python
"""The chain's queue, configs/chain/runs.yaml: read, filled in, and checked.

Checked hard, because the first place a bad entry would otherwise surface is a
Kaggle session ten minutes in -- or never, if the typo is in a field only a
later session reads.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from ..config import ConfigError

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
# Kaggle titles are at most 50 characters; the kernel is "deluge-<name>-a".
MAX_NAME = 50 - len("deluge-") - len("-a")
ACCELERATORS = ("cpu", "gpu")
DATA_RE = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+:[^\s:]+$")
REQUIRED = ("name", "model", "train", "accelerator", "data")
DEFAULTS = {"model_impl": "deluge.model:build"}


def load_queue(path: Union[str, Path]) -> List[Dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    runs = raw.get("runs") or []
    if not isinstance(runs, list):
        raise ConfigError(f"{path}: `runs` must be a list")

    seen, queue = set(), []
    for index, entry in enumerate(runs):
        where = f"{path}: runs[{index}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where} must be a mapping")
        missing = [key for key in REQUIRED if key not in entry]
        if missing:
            raise ConfigError(f"{where} is missing {', '.join(missing)}")
        unknown = sorted(set(entry) - set(REQUIRED) - set(DEFAULTS))
        if unknown:
            raise ConfigError(f"{where} has unknown field(s) {', '.join(unknown)}")

        name = entry["name"]
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise ConfigError(
                f"{where}: name {name!r} must be lowercase letters, digits and "
                f"single hyphens -- it becomes the Kaggle kernel slug")
        if len(name) > MAX_NAME:
            raise ConfigError(f"{where}: name {name!r} is over {MAX_NAME} characters")
        if name in seen:
            raise ConfigError(f"{where}: name {name!r} appears more than once")
        seen.add(name)
        if entry["accelerator"] not in ACCELERATORS:
            raise ConfigError(
                f"{where}: accelerator {entry['accelerator']!r} must be one of "
                f"{', '.join(ACCELERATORS)}")
        data = entry["data"]
        if data != "synthetic" and not (isinstance(data, str) and DATA_RE.match(data)):
            raise ConfigError(
                f"{where}: data {data!r} must be `synthetic` or "
                f"`<owner>/<dataset>:<file>`")
        queue.append({**DEFAULTS, **entry})
    return queue
```

`deluge/train/smoke.py`:

```python
"""Test-only model for the chain rehearsal (configs/chain/runs.yaml: smoke).

Not a language model worth training: an embedding and a linear head, just
enough to drive the trainer's real torch path -- AdamW, checkpoints, resume --
on Kaggle's CPU. The sleep makes a step's wall clock independent of whatever
CPU Kaggle hands out, which is what lets configs/train/smoke.yaml promise
exactly three sessions. build_crashing is the forced-failure arm: it raises
before the first step, which is what the chain must halt on.
"""

import time

import torch
from torch import nn
from torch.nn import functional as F

SECONDS_PER_STEP = 0.5
WIDTH = 64


class SmokeLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, WIDTH)
        self.head = nn.Linear(WIDTH, vocab_size)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        time.sleep(SECONDS_PER_STEP)
        logits = self.head(self.embed(inputs.long()))
        return F.cross_entropy(logits.flatten(0, 1), targets.long().flatten())


def build(model_cfg) -> SmokeLM:
    return SmokeLM(model_cfg.vocab_size)


def build_crashing(model_cfg):
    raise RuntimeError("smoke-crash: deliberate failure, for the chain's halt path")
```

`configs/train/smoke.yaml`:

```yaml
# Chain rehearsal budget, paired with deluge/train/smoke.py -- not a model run.
#
# smoke.py sleeps 0.5 s per forward and grad_accum is 1, so a step takes
# 0.5-0.7 s whatever CPU Kaggle provides: a 2-minute deadline fits 171-240
# steps, and 500 steps therefore takes exactly three sessions -- enough to
# exercise a -> b -> a, resume, and completion. tests/test_chain_queue.py
# holds that arithmetic.
extends: base.yaml

seq_len: 64
global_batch_tokens: 256          # 4 sequences of 64
micro_batch: 4                    # grad_accum 1: one forward, one sleep, per step
total_tokens: 128_000             # 500 steps
warmup_tokens: 2_560
checkpoint_every_tokens: 12_800   # every 50 steps
dtype: fp32                       # CPU
deadline_minutes: 2
```

`configs/chain/runs.yaml`:

```yaml
# The chain's queue. Runs are worked through in order, one at a time, each
# over as many ~11 h Kaggle sessions as its budget needs. docs/kaggle.md
# ("The chain") says how to add one; the design is
# docs/superpowers/specs/2026-09-27-kaggle-chain-design.md.
#
# An entry is snapshotted when its run starts, so editing it afterwards only
# matters for runs not yet started. Names are also identities: a name already
# in .github/chain/state.json's `done` is never run again -- rename to rerun.
#
#   name:        [a-z0-9-], becomes the kernels deluge-<name>-a / -b
#   model/train: config paths from the repo root
#   model_impl:  factory, default deluge.model:build
#   accelerator: cpu | gpu (gpu = Kaggle T4)
#   data:        synthetic | <owner>/<dataset>:<file>
runs:
  # Rehearsal: tiny stub model on Kaggle CPU, no GPU quota, three sessions.
  - name: smoke
    model: configs/model/screen.yaml
    train: configs/train/smoke.yaml
    model_impl: deluge.train.smoke:build
    accelerator: cpu
    data: synthetic
```

- [ ] **Step 4: Run to verify they pass**

Run: `$PY -m pytest tests/test_chain_queue.py -v`
Expected: all PASS. If `test_smoke_budget_needs_exactly_three_sessions` fails, fix `total_tokens` in `smoke.yaml`, not the test.

- [ ] **Step 5: Commit**

```bash
git add deluge/chain/__init__.py deluge/chain/queue.py deluge/train/smoke.py \
        configs/chain/runs.yaml configs/train/smoke.yaml tests/test_chain_queue.py
git commit -m "chain: the run queue, and a three-session smoke run to rehearse on

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The decision function

**Files:**
- Create: `deluge/chain/step.py`
- Test: `tests/test_chain_step.py`

**Interfaces:**
- Consumes: queue entries from `load_queue` (Task 2).
- Produces:
  - `initial_state() -> dict`
  - `step(state, queue, observation, now, command="tick", commit=None) -> (dict, Action)`
  - `api_error(state, reason) -> (dict, Action)`
  - `push_refused(state, now) -> dict`
  - `kernel_slug(name, side) -> str` and `other(side) -> str`
  - `Observation(status: str, session: Optional[dict])`
  - `Push(side: str, first: bool)` and `Fail(reason: str)`; `Action = None | Push | Fail`
  - constants `DONE=0, RESUMABLE=2, NO_GPU=3, GRACE, STUCK_AFTER, WAIT_FOR, MAX_CRASHES=2, MAX_API_ERRORS=6`
- State keys, exactly: `status, run, side, session, pushed_at, handed_off, last_tokens, crash_streak, api_errors, wait_until, next_side, done`. Times are ISO-8601 strings with an offset. `now` is a tz-aware `datetime`.

- [ ] **Step 1: Write the failing tests**

`tests/test_chain_step.py`:

```python
"""Every transition in the chain design's two tables, one test each.

step() is pure, so these need no Kaggle, no clock and no GitHub.
"""

from datetime import datetime, timedelta, timezone

import pytest

from deluge.chain.step import (
    GRACE, MAX_API_ERRORS, STUCK_AFTER, WAIT_FOR, Fail, Observation, Push,
    api_error, initial_state, kernel_slug, push_refused, step,
)

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
RUN = {"name": "smoke", "model": "m.yaml", "train": "t.yaml",
       "model_impl": "deluge.train.smoke:build", "accelerator": "cpu",
       "data": "synthetic"}


def iso(t):
    return t.isoformat(timespec="seconds")


def running(side="a", session=1, handed_off=False, pushed=NOW - timedelta(hours=2),
            **overrides):
    state = initial_state()
    state.update(status="running", side=side, session=session,
                 handed_off=handed_off, pushed_at=iso(pushed),
                 run={**RUN, "commit": "abc123", "run_id": "smoke@2026-09-27T00:00:00+00:00"})
    state.update(overrides)
    return state


def report(state, exit_code, tokens, status="complete", **overrides):
    session = {"run_id": state["run"]["run_id"], "session": state["session"],
               "side": state["side"], "exit_code": exit_code,
               "tokens_seen": tokens, "commit": "abc123", **overrides}
    return Observation(status=status, session=session)


# ---- starting ------------------------------------------------------------ #

def test_idle_starts_the_first_queued_run_on_a_with_no_sources():
    state, action = step(initial_state(), [RUN], None, NOW, commit="abc123")
    assert action == Push(side="a", first=True)
    assert state["status"] == "running" and state["session"] == 1
    assert state["run"]["commit"] == "abc123"
    assert state["run"]["run_id"] == f"smoke@{iso(NOW)}"
    assert state["pushed_at"] == iso(NOW)


def test_idle_skips_runs_already_done():
    start = initial_state()
    start["done"] = [{"name": "smoke"}]
    later = {**RUN, "name": "a0"}
    state, action = step(start, [RUN, later], None, NOW, commit="abc123")
    assert action == Push(side="a", first=True) and state["run"]["name"] == "a0"


def test_idle_with_nothing_queued_does_nothing():
    start = initial_state()
    start["done"] = [{"name": "smoke"}]
    state, action = step(start, [RUN], None, NOW, commit="abc123")
    assert action is None and state == start


def test_starting_a_run_needs_a_commit_to_pin():
    with pytest.raises(ValueError, match="commit"):
        step(initial_state(), [RUN], None, NOW)


def test_step_does_not_mutate_its_input():
    start = running()
    before = repr(start)
    step(start, [RUN], report(start, 2, 100), NOW)
    assert repr(start) == before


# ---- the normal path ------------------------------------------------------ #

def test_no_decision_within_grace_after_push():
    # Review focus 1: the status may still be the previous version's.
    start = running(pushed=NOW - GRACE + timedelta(seconds=1))
    state, action = step(start, [RUN], report(start, 2, 100), NOW)
    assert action is None and state == start


@pytest.mark.parametrize("status", ["queued", "running"])
def test_a_live_session_is_left_alone(status):
    start = running()
    state, action = step(start, [RUN], Observation(status, None), NOW)
    assert action is None and state == start


def test_resumable_exit_hands_off_to_the_other_side():
    start = running(crash_streak=1)
    state, action = step(start, [RUN], report(start, 2, 5000), NOW)
    assert action == Push(side="b", first=False)
    assert state["side"] == "b" and state["session"] == 2
    assert state["last_tokens"] == 5000 and state["crash_streak"] == 0
    assert state["handed_off"] is True


def test_completion_records_the_run_and_goes_idle():
    start = running(side="a", session=3, handed_off=True, last_tokens=900)
    state, action = step(start, [RUN], report(start, 0, 1000), NOW)
    assert action is None and state["status"] == "idle" and state["run"] is None
    [done] = state["done"]
    assert done["name"] == "smoke" and done["sessions"] == 3
    assert done["commit"] == "abc123" and done["finished"] == iso(NOW)


# ---- failures -------------------------------------------------------------- #

def test_training_crash_hands_off_and_counts():
    start = running(last_tokens=100)
    state, action = step(start, [RUN], report(start, 1, 100), NOW)
    assert action == Push(side="b", first=False)
    assert state["crash_streak"] == 1


def test_second_crash_without_progress_halts():
    start = running(side="b", handed_off=True, crash_streak=1, last_tokens=100)
    state, action = step(start, [RUN], report(start, 1, 100), NOW)
    assert isinstance(action, Fail) and "smoke" in action.reason
    assert state["status"] == "halted" and state["next_side"] == "a"


def test_a_crash_after_progress_starts_the_count_again():
    start = running(crash_streak=1, last_tokens=100)
    state, action = step(start, [RUN], report(start, 1, 200), NOW)
    assert isinstance(action, Push) and state["crash_streak"] == 1


def test_exit_2_without_progress_counts_as_a_failure():
    # Review focus 2: a session that trains nothing must not loop forever.
    start = running(last_tokens=100)
    state, action = step(start, [RUN], report(start, 2, 100), NOW)
    assert action == Push(side="b", first=False) and state["crash_streak"] == 1


@pytest.mark.parametrize("status", ["error", "cancelled", "unknown", "complete"])
def test_kaggle_kill_without_session_json_retries_the_same_side(status):
    start = running(side="a", session=1, handed_off=False)
    state, action = step(start, [RUN], Observation(status, None), NOW)
    # Same side: its input is unchanged. Still first: nothing has handed off.
    assert action == Push(side="a", first=True)
    assert state["session"] == 2 and state["crash_streak"] == 1


def test_a_stale_session_json_is_ignored():
    start = running(session=4)
    stale = report(start, 2, 100, session=3)
    state, action = step(start, [RUN], stale, NOW)
    assert action == Push(side="a", first=True) and state["crash_streak"] == 1


def test_a_session_json_from_an_earlier_run_is_ignored():
    start = running()
    stale = report(start, 0, 100, run_id="smoke@2026-01-01T00:00:00+00:00")
    _, action = step(start, [RUN], stale, NOW)
    assert action == Push(side="a", first=True)


def test_stuck_session_is_treated_as_a_kill():
    start = running(pushed=NOW - STUCK_AFTER - timedelta(minutes=1))
    state, action = step(start, [RUN], Observation("running", None), NOW)
    assert action == Push(side="a", first=True) and state["crash_streak"] == 1


def test_no_gpu_waits_without_counting_a_crash():
    start = running(side="b", handed_off=True)
    state, action = step(start, [RUN], report(start, 3, 0), NOW)
    assert action is None and state["status"] == "waiting"
    assert state["wait_until"] == iso(NOW + WAIT_FOR)
    assert state["next_side"] == "b" and state["crash_streak"] == 0


def test_waiting_until_the_backoff_ends_then_pushing():
    start = running(side="b", handed_off=True)
    start.update(status="waiting", next_side="b", wait_until=iso(NOW + timedelta(hours=1)))
    state, action = step(start, [RUN], None, NOW)
    assert action is None and state == start
    state, action = step(start, [RUN], None, NOW + timedelta(hours=1))
    assert action == Push(side="b", first=False) and state["status"] == "running"
    assert state["wait_until"] is None and state["next_side"] is None


def test_push_refused_for_quota_waits_on_the_side_it_tried():
    start = running(side="b", session=2, handed_off=True)
    state = push_refused(start, NOW)
    assert state["status"] == "waiting" and state["next_side"] == "b"
    assert state["wait_until"] == iso(NOW + WAIT_FOR)


# ---- halted, commands, API errors ------------------------------------------- #

def test_a_halted_chain_stays_quiet_on_tick():
    start = running(status="halted", next_side="a", crash_streak=2)
    state, action = step(start, [RUN], None, NOW)
    assert action is None and state == start


def test_resume_restarts_a_halted_chain_on_the_side_it_would_have_pushed():
    start = running(status="halted", next_side="b", crash_streak=2, handed_off=True)
    state, action = step(start, [RUN], None, NOW, command="resume")
    assert action == Push(side="b", first=False)
    assert state["crash_streak"] == 0 and state["status"] == "running"


def test_resume_on_an_idle_chain_does_nothing():
    state, action = step(initial_state(), [RUN], None, NOW, command="resume")
    assert action is None and state == initial_state()


def test_skip_drops_the_current_run():
    state, action = step(running(), [RUN], None, NOW, command="skip")
    assert action is None and state["status"] == "idle"
    assert state["done"][-1]["name"] == "smoke" and state["done"][-1]["skipped"] is True


def test_skip_with_no_current_run_does_nothing():
    state, action = step(initial_state(), [RUN], None, NOW, command="skip")
    assert action is None and state == initial_state()


def test_api_errors_fail_once_at_the_threshold():
    state, actions = running(), []
    for _ in range(MAX_API_ERRORS + 1):
        state, action = api_error(state, "timeout")
        actions.append(action)
    assert actions[:MAX_API_ERRORS - 1] == [None] * (MAX_API_ERRORS - 1)
    assert isinstance(actions[MAX_API_ERRORS - 1], Fail)
    assert actions[MAX_API_ERRORS] is None      # one email, not one per tick
    assert state["status"] == "running"          # an outage is not a halt


def test_a_successful_step_clears_the_api_error_count():
    start = running(api_errors=3)
    state, _ = step(start, [RUN], Observation("running", None), NOW)
    assert state["api_errors"] == 0


def test_kernel_slug():
    assert kernel_slug("a0-scan-only", "b") == "deluge-a0-scan-only-b"
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_chain_step.py -v`
Expected: collection error, `No module named 'deluge.chain.step'`.

- [ ] **Step 3: Implement `deluge/chain/step.py`**

```python
"""The chain's decision: given where it is and what Kaggle reports, what next.

Pure -- no I/O, no clock, no Kaggle -- so every row of the design's two
transition tables is a unit test (tests/test_chain_step.py). orchestrate.py
does the observing and acting around it.

The state is a plain dict because it is a JSON file in the repo
(.github/chain/state.json); initial_state() is its shape.
"""

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# Exit codes a session reports. 0 and 2 are deluge.train's (Outcome.exit_code);
# 3 is session.py's own "no usable GPU", which is not the run's fault.
DONE, RESUMABLE, NO_GPU = 0, 2, 3

# A status read this soon after a push may still describe the previous version
# of the kernel -- and that version's session.json is right there to misread.
GRACE = timedelta(minutes=10)
# Kaggle kills a session at 12 h; the extra hour is queueing slack.
STUCK_AFTER = timedelta(hours=13)
# GPU quota is weekly; retrying sooner than this only burns API calls.
WAIT_FOR = timedelta(hours=6)
MAX_CRASHES = 2
MAX_API_ERRORS = 6                  # ~3 h of 30-minute ticks

State = Dict[str, Any]


@dataclass(frozen=True)
class Observation:
    status: str                     # queued|running|complete|error|cancelled|unknown
    session: Optional[Dict[str, Any]]   # the kernel's session.json, if any


@dataclass(frozen=True)
class Push:
    side: str
    first: bool                     # no kernel_sources: nothing has handed off yet


@dataclass(frozen=True)
class Fail:
    reason: str


Action = Union[None, Push, Fail]


def initial_state() -> State:
    return {"status": "idle", "run": None, "side": None, "session": 0,
            "pushed_at": None, "handed_off": False, "last_tokens": 0,
            "crash_streak": 0, "api_errors": 0, "wait_until": None,
            "next_side": None, "done": []}


def kernel_slug(name: str, side: str) -> str:
    return f"deluge-{name}-{side}"


def other(side: str) -> str:
    return "b" if side == "a" else "a"


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


# ---- building blocks ------------------------------------------------------- #

def _push(s: State, side: str, now: datetime) -> Tuple[State, Action]:
    """Every push gets a fresh session number: (run_id, session) names it."""
    s.update(status="running", side=side, session=s["session"] + 1,
             pushed_at=_iso(now), wait_until=None, next_side=None)
    return s, Push(side=side, first=not s["handed_off"])


def _wait(s: State, side: str, now: datetime) -> State:
    s.update(status="waiting", next_side=side, wait_until=_iso(now + WAIT_FOR))
    return s


def _failed(s: State, next_side: str, now: datetime, why: str) -> Tuple[State, Action]:
    s["crash_streak"] += 1
    if s["crash_streak"] >= MAX_CRASHES:
        s.update(status="halted", next_side=next_side)
        return s, Fail(f"{s['run']['name']}: {why} -- {s['crash_streak']} failures "
                       f"in a row without progress. Fix it, then run the chain "
                       f"workflow with command=resume (or skip).")
    return _push(s, next_side, now)


def _record(s: State, now: datetime, skipped: bool = False) -> Dict[str, Any]:
    record = {"name": s["run"]["name"], "run_id": s["run"]["run_id"],
              "commit": s["run"]["commit"], "sessions": s["session"],
              "finished": _iso(now)}
    if skipped:
        record["skipped"] = True
    return record


def _idle(s: State) -> State:
    done = s["done"]
    s.clear()
    s.update(initial_state(), done=done)
    return s


def _fresh(session: Optional[Dict[str, Any]], s: State) -> Optional[Dict[str, Any]]:
    """session.json only counts if it is this push's -- not an earlier one's."""
    if (isinstance(session, dict)
            and session.get("run_id") == s["run"]["run_id"]
            and session.get("session") == s["session"]):
        return session
    return None


# ---- the decision ------------------------------------------------------------ #

def step(state: State, queue: List[Dict[str, Any]], observation: Optional[Observation],
         now: datetime, command: str = "tick",
         commit: Optional[str] = None) -> Tuple[State, Action]:
    s = copy.deepcopy(state)
    s["api_errors"] = 0             # reaching step() means Kaggle answered, or wasn't asked

    if command == "skip":
        if s["run"] is None:
            return s, None
        s["done"].append(_record(s, now, skipped=True))
        return _idle(s), None
    if command == "resume":
        if s["status"] in ("halted", "waiting"):
            s["crash_streak"] = 0
            return _push(s, s["next_side"], now)
        return s, None

    status = s["status"]
    if status == "halted":
        return s, None
    if status == "waiting":
        if now >= datetime.fromisoformat(s["wait_until"]):
            return _push(s, s["next_side"], now)
        return s, None
    if status == "idle":
        return _start(s, queue, now, commit)
    if observation is None:
        raise ValueError("a running chain needs an observation of its kernel")
    return _running(s, observation, now)


def _start(s: State, queue, now, commit) -> Tuple[State, Action]:
    finished = {record["name"] for record in s["done"]}
    pending = [run for run in queue if run["name"] not in finished]
    if not pending:
        return s, None
    if not commit:
        raise ValueError("starting a run needs the commit to pin it to")
    run = pending[0]
    s.update(run={**run, "commit": commit, "run_id": f"{run['name']}@{_iso(now)}"},
             session=0, handed_off=False, last_tokens=0, crash_streak=0)
    return _push(s, "a", now)


def _running(s: State, observation: Observation, now: datetime) -> Tuple[State, Action]:
    pushed = datetime.fromisoformat(s["pushed_at"])
    if now - pushed < GRACE:
        return s, None
    side = s["side"]

    if observation.status in ("queued", "running"):
        if now - pushed > STUCK_AFTER:
            return _failed(s, side, now, f"kernel still {observation.status} after "
                                         f"{STUCK_AFTER}")
        return s, None

    session = _fresh(observation.session, s)
    if session is None:
        # Killed by Kaggle, or ended without writing session.json. Re-push the
        # same side: its input -- the other side's output -- still holds the
        # last good checkpoint, whatever this side's output did or did not keep.
        return _failed(s, side, now, f"kernel ended {observation.status} "
                                     f"without a session.json")

    code = session.get("exit_code")
    if code == NO_GPU:
        return _wait(s, side, now), None
    # A fresh session.json means this side's output exists and carries the
    # newest checkpoint (session.py copies it forward first), so hand off.
    s["handed_off"] = True
    tokens = session.get("tokens_seen") or 0
    progressed = tokens > s["last_tokens"]
    if progressed:
        s.update(crash_streak=0, last_tokens=tokens)
    if code == DONE:
        s["done"].append(_record(s, now))
        return _idle(s), None
    if code == RESUMABLE and progressed:
        return _push(s, other(side), now)
    why = ("session ended resumable but trained nothing" if code == RESUMABLE
           else f"training exited {code}")
    return _failed(s, other(side), now, why)


def api_error(state: State, reason: str) -> Tuple[State, Action]:
    """Kaggle did not answer. Change nothing else; fail once at the threshold."""
    s = copy.deepcopy(state)
    s["api_errors"] += 1
    if s["api_errors"] == MAX_API_ERRORS:
        return s, Fail(f"Kaggle API failing for {MAX_API_ERRORS} ticks in a row: {reason}")
    return s, None


def push_refused(state: State, now: datetime) -> State:
    """Kaggle refused a push for quota: wait, then retry the same push."""
    return _wait(copy.deepcopy(state), state["side"], now)
```

- [ ] **Step 4: Run to verify they pass**

Run: `$PY -m pytest tests/test_chain_step.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add deluge/chain/step.py tests/test_chain_step.py
git commit -m "chain: the decision function, with a test per transition

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The session script, and `Checkpointer.newest_readable`

**Files:**
- Modify: `deluge/train/checkpoint.py`: add a method after `candidates`, around line 171
- Create: `deluge/chain/session.py`
- Test: `tests/test_checkpoint.py` (append), `tests/test_chain_session.py`

**Interfaces:**
- Consumes: `NO_GPU` from `deluge.chain.step` (asserted equal by a test; `session.py` can't import it because it runs before the package is installed).
- Produces: `Checkpointer.newest_readable(search_dirs=()) -> Optional[tuple[int, Path]]`.
- Produces in `session.py`:
  - `RUN_MARKER = "RUN = {}"` (the exact line prefix `render()` replaces)
  - `main(params=None, work=WORK, input_root=INPUT, src=SRC, run_cmd=subprocess.run, gpus=usable_gpus) -> int`
  - `params` shape: `{"run": <queue entry + commit + run_id>, "session": int, "side": "a"|"b", "repo": str}`
  - it writes `work/session.json` with keys `run_id, session, side, commit, exit_code, tokens_seen`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_checkpoint.py`:

```python
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
```

`tests/test_chain_session.py`:

```python
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
    session.main(params(), work=work, input_root=tmp_path / "in", src=tmp_path / "src",
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_checkpoint.py tests/test_chain_session.py -v`
Expected: `AttributeError: 'Checkpointer' object has no attribute 'newest_readable'` and `ImportError: cannot import name 'session'`.

- [ ] **Step 3: Add `newest_readable` to `deluge/train/checkpoint.py`, directly after `candidates`**

```python
    def newest_readable(self, search_dirs: Iterable[Path] = ()) -> Optional[tuple[int, Path]]:
        """The newest checkpoint that deserializes, without restoring it.

        What a new session copies forward (deluge/chain/session.py). It uses the
        same discovery and the same loader as load_latest, so "the checkpoint the
        next session will resume from" cannot drift between the two.
        """
        for step, path in self.candidates(search_dirs):
            try:
                _load(path)
            except Exception:  # noqa: BLE001 - any failure means "try older"
                continue
            return step, path
        return None
```

- [ ] **Step 4: Create `deluge/chain/session.py`**

```python
"""One Kaggle session of a chained run. Pushed as the kernel's code file.

deluge.chain.kaggle.render writes this session's parameters into RUN below and
pushes the file; Kaggle runs it top to bottom. Until step 2 installs the
package it may use only the standard library and torch (which Kaggle images
carry), so nothing here imports deluge at module level.

  1. no usable GPU on a gpu run -> report 3 and stop (quota, or a P100)
  2. clone the pinned commit, pip install -e
  3. copy the other side's newest readable checkpoint and log into this output
  4. python -m deluge.train
  5. always: write /kaggle/working/session.json, exit 0

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
    try:
        found = gpus()
        if run["accelerator"] == "gpu" and found == 0:
            print("[session] no CUDA device of capability >= 7.0; reporting 3")
            record["exit_code"] = NO_GPU
            return 0
        fetch(params["repo"], run["commit"], src, run_cmd)
        carried = carry_forward(find_prior_run_dir(input_root, run["name"]), out_dir)
        print(f"[session] {run['run_id']} session {params['session']} on "
              f"{params['side']}, resuming from {carried}")
        data = resolve_data(run["data"], input_root, src.parent)
        result = run_cmd(train_command(run, data, out_dir, found, src), cwd=str(src))
        record["exit_code"] = result.returncode
    except Exception:  # noqa: BLE001 - anything at all must still reach session.json
        traceback.print_exc()
    finally:
        record["tokens_seen"] = tokens_seen(out_dir / "log.jsonl")
        work.mkdir(parents=True, exist_ok=True)
        (work / "session.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"[session] {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`resolve_data` writes synthetic tokens beside the clone (`/tmp` on Kaggle, `tmp_path` in tests), never under `work`, so they never become part of the kernel's output.

- [ ] **Step 5: Run to verify they pass**

Run: `$PY -m pytest tests/test_checkpoint.py tests/test_chain_session.py -v`
Expected: all PASS. If Task 1 found a mount layout the `**/runs/<name>` glob can't reach, adjust `find_prior_run_dir` and add a test case with that exact layout.

- [ ] **Step 6: Commit**

```bash
git add deluge/train/checkpoint.py deluge/chain/session.py \
        tests/test_checkpoint.py tests/test_chain_session.py
git commit -m "chain: the session script, carrying the newest checkpoint forward

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The Kaggle adapter

**Files:**
- Create: `deluge/chain/kaggle.py`
- Test: `tests/test_chain_kaggle.py`

**Interfaces:**
- Consumes: `Push`, `kernel_slug`, `other` (Task 3); `RUN_MARKER` and the source file of `deluge/chain/session.py` (Task 4).
- Produces:
  - exceptions `KaggleError(RuntimeError)` and `KaggleQuotaError(KaggleError)`
  - `render(dest: Path, owner: str, state: dict, action: Push, session_source: Path = SESSION_PY) -> None`, which writes `dest/session.py` and `dest/kernel-metadata.json`
  - `KaggleCLI(owner: str, runner=subprocess.run, binary="kaggle")` with methods `.status(slug) -> str`, `.fetch_session_json(slug) -> Optional[dict]` and `.push(state, action) -> None`

- [ ] **Step 1: Write the failing tests**

`tests/test_chain_kaggle.py`:

```python
"""The kaggle CLI adapter, against a fake runner: no network."""

import json
import subprocess
from pathlib import Path

import pytest

from deluge.chain.kaggle import KaggleCLI, KaggleError, KaggleQuotaError, render
from deluge.chain.step import Push

RUN = {"name": "smoke", "model": "m.yaml", "train": "t.yaml",
       "model_impl": "deluge.train.smoke:build", "accelerator": "gpu",
       "data": "heylain/deluge-tokens:tokens.bin",
       "commit": "abc123", "run_id": "smoke@t0"}
STATE = {"run": RUN, "session": 3, "side": "b"}


class FakeRunner:
    def __init__(self, stdout="", returncode=0, stderr="", write=None):
        self.stdout, self.returncode, self.stderr, self.write = stdout, returncode, stderr, write
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self.write and "-p" in cmd:
            self.write(Path(cmd[cmd.index("-p") + 1]))
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def metadata(tmp_path, action, state=STATE):
    render(tmp_path, "heylain", state, action)
    return json.loads((tmp_path / "kernel-metadata.json").read_text())


# ---- render ---------------------------------------------------------------- #

def test_first_push_has_no_kernel_sources_and_asks_for_a_t4(tmp_path):
    meta = metadata(tmp_path, Push("a", first=True), {**STATE, "side": "a"})
    assert meta["id"] == "heylain/deluge-smoke-a" and meta["title"] == "deluge-smoke-a"
    assert meta["kernel_sources"] == []
    assert meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["is_private"] is True and meta["enable_internet"] is True
    assert meta["dataset_sources"] == ["heylain/deluge-tokens"]


def test_handoff_sources_the_other_side(tmp_path):
    meta = metadata(tmp_path, Push("b", first=False))
    assert meta["kernel_sources"] == ["heylain/deluge-smoke-a"]


def test_cpu_synthetic_run_asks_for_nothing(tmp_path):
    state = {**STATE, "run": {**RUN, "accelerator": "cpu", "data": "synthetic"}}
    meta = metadata(tmp_path, Push("b", first=False), state)
    assert meta["enable_gpu"] is False and meta["machine_shape"] == ""
    assert meta["dataset_sources"] == []


def test_rendered_session_carries_this_push_identity(tmp_path):
    render(tmp_path, "heylain", STATE, Push("b", first=False))
    namespace = {"__name__": "rendered"}
    exec(compile((tmp_path / "session.py").read_text(), "session.py", "exec"), namespace)
    assert namespace["RUN"]["run"] == RUN
    assert namespace["RUN"]["session"] == 3 and namespace["RUN"]["side"] == "b"
    assert namespace["RUN"]["repo"] == "https://github.com/heylain/Deluge"


def test_render_refuses_a_session_source_without_the_marker(tmp_path):
    source = tmp_path / "session.py"
    source.write_text("print('no marker here')\n")
    with pytest.raises(KaggleError, match="RUN = {}"):
        render(tmp_path / "out", "heylain", STATE, Push("b", False), session_source=source)


# ---- status ------------------------------------------------------------------ #

@pytest.mark.parametrize("raw, expected", [
    ("KernelWorkerStatus.COMPLETE", "complete"),
    ("complete", "complete"),
    ("KernelWorkerStatus.RUNNING", "running"),
    ("queued", "queued"),
    ("ERROR", "error"),
    ("cancelAcknowledged", "cancelled"),
    ("KernelWorkerStatus.CANCEL_REQUESTED", "cancelled"),
    ("NEW_THING", "unknown"),
])
def test_status_parses_enum_and_plain_forms(raw, expected):
    # Review focus 5.
    runner = FakeRunner(stdout=f'heylain/deluge-smoke-a has status "{raw}"\n')
    assert KaggleCLI("heylain", runner=runner).status("deluge-smoke-a") == expected
    assert runner.calls[0][-1] == "heylain/deluge-smoke-a"


def test_unparseable_status_raises():
    with pytest.raises(KaggleError, match="status"):
        KaggleCLI("heylain", runner=FakeRunner(stdout="???")).status("x")


def test_a_failing_cli_raises_and_quota_is_its_own_error():
    with pytest.raises(KaggleError):
        KaggleCLI("heylain", runner=FakeRunner(returncode=1, stderr="503")).status("x")
    with pytest.raises(KaggleQuotaError):
        KaggleCLI("heylain", runner=FakeRunner(
            returncode=1, stderr="GPU quota exceeded")).status("x")


# ---- fetch ------------------------------------------------------------------- #

def test_fetch_session_json_reads_only_that_file():
    runner = FakeRunner(write=lambda d: (d / "session.json").write_text('{"exit_code": 2}'))
    assert KaggleCLI("heylain", runner=runner).fetch_session_json("s") == {"exit_code": 2}
    assert "--file-pattern" in runner.calls[0]


@pytest.mark.parametrize("write", [None, lambda d: (d / "session.json").write_text("{trunc")])
def test_missing_or_corrupt_session_json_is_none(write):
    assert KaggleCLI("heylain", runner=FakeRunner(write=write)).fetch_session_json("s") is None


# ---- push -------------------------------------------------------------------- #

def test_push_renders_and_pushes():
    seen = {}
    runner = FakeRunner(stdout="Kernel version 4 successfully pushed.",
                        write=lambda d: seen.update(files=sorted(p.name for p in d.iterdir())))
    KaggleCLI("heylain", runner=runner).push(STATE, Push("b", first=False))
    assert runner.calls[0][:3] == ["kaggle", "kernels", "push"]
    assert seen["files"] == ["kernel-metadata.json", "session.py"]


def test_push_without_success_line_raises():
    # Review focus 4: the CLI has exited 0 while printing an error.
    runner = FakeRunner(stdout="Kernel push error: Notebook not found")
    with pytest.raises(KaggleError, match="Notebook not found"):
        KaggleCLI("heylain", runner=runner).push(STATE, Push("b", first=False))


def test_push_refused_for_quota_is_a_quota_error():
    runner = FakeRunner(stdout="Kernel push error: You have exceeded your GPU quota")
    with pytest.raises(KaggleQuotaError):
        KaggleCLI("heylain", runner=runner).push(STATE, Push("b", first=False))
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_chain_kaggle.py -v`
Expected: `ImportError: cannot import name 'KaggleCLI'`.

- [ ] **Step 3: Implement `deluge/chain/kaggle.py`**

Apply Task 1's findings to `STATUS_ALIASES`, `PUSH_OK` and `MACHINE_SHAPE` if they differed.

```python
"""The kaggle CLI, wrapped: status, fetch session.json, push a session.

Thin on purpose. Everything that decides lives in step.py; this module turns
CLI output into step.py's vocabulary and raises one error type for "Kaggle did
not do it", with quota refusals split out because the chain waits on those
instead of counting them. Constants here were checked against the real CLI in
docs/kaggle-chain-spike.md.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from .session import RUN_MARKER
from .step import Push, kernel_slug, other

SESSION_PY = Path(__file__).with_name("session.py")
REPO = "https://github.com/heylain/Deluge"
MACHINE_SHAPE = "NvidiaTeslaT4"     # the P100 cannot run Triton (sm_60)
PUSH_OK = "successfully pushed"
STATUS_RE = re.compile(r'status "([^"]+)"')
STATUS_ALIASES = {
    "queued": "queued", "running": "running", "complete": "complete",
    "error": "error", "cancelrequested": "cancelled",
    "cancelacknowledged": "cancelled", "cancelled": "cancelled",
}


class KaggleError(RuntimeError):
    """Kaggle did not do what was asked. The chain retries on the next tick."""


class KaggleQuotaError(KaggleError):
    """Kaggle refused for quota. The chain waits rather than counting a failure."""


def _raise_for(text: str) -> None:
    error = KaggleQuotaError if "quota" in text.lower() else KaggleError
    raise error(text.strip() or "kaggle CLI failed with no output")


def render(dest: Path, owner: str, state: dict, action: Push,
           session_source: Path = SESSION_PY) -> None:
    """Write the kernel directory `kaggle kernels push -p` expects."""
    run = state["run"]
    source = session_source.read_text()
    marker = re.compile(rf"^{re.escape(RUN_MARKER)}.*$", re.MULTILINE)
    if not marker.search(source):
        raise KaggleError(f"{session_source} has no line starting {RUN_MARKER!r} to render")
    params = {"run": run, "session": state["session"], "side": action.side, "repo": REPO}
    # json inside a Python string literal: JSON's true/null are not Python.
    rendered = marker.sub(
        lambda _: f"RUN = json.loads({json.dumps(json.dumps(params))})", source, count=1)

    gpu = run["accelerator"] == "gpu"
    slug = kernel_slug(run["name"], action.side)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "session.py").write_text(rendered)
    (dest / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": "session.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": True,        # the session clones the repo
        "enable_gpu": gpu,
        "machine_shape": MACHINE_SHAPE if gpu else "",
        "dataset_sources": [] if run["data"] == "synthetic" else [run["data"].split(":")[0]],
        "competition_sources": [],
        "kernel_sources": [] if action.first
                          else [f"{owner}/{kernel_slug(run['name'], other(action.side))}"],
    }, indent=2) + "\n")


class KaggleCLI:
    def __init__(self, owner: str, runner: Callable = subprocess.run, binary: str = "kaggle"):
        self.owner, self.runner, self.binary = owner, runner, binary

    def _run(self, *args: str) -> str:
        result = self.runner([self.binary, *args], capture_output=True, text=True)
        if result.returncode != 0:
            _raise_for(f"{result.stderr}\n{result.stdout}")
        return result.stdout

    def status(self, slug: str) -> str:
        out = self._run("kernels", "status", f"{self.owner}/{slug}")
        match = STATUS_RE.search(out)
        if not match:
            raise KaggleError(f"cannot read a status from {out!r}")
        # "KernelWorkerStatus.CANCEL_REQUESTED" and "cancelRequested" alike
        raw = match.group(1).rsplit(".", 1)[-1].replace("_", "").lower()
        return STATUS_ALIASES.get(raw, "unknown")

    def fetch_session_json(self, slug: str) -> Optional[dict]:
        with tempfile.TemporaryDirectory() as tmp:
            self._run("kernels", "output", f"{self.owner}/{slug}", "-p", tmp,
                      "--file-pattern", r"^session\.json$")
            path = Path(tmp) / "session.json"
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return None

    def push(self, state: dict, action: Push) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            render(Path(tmp), self.owner, state, action)
            out = self._run("kernels", "push", "-p", tmp)
        if PUSH_OK not in out.lower():
            _raise_for(out)
```

The `FakeRunner` in `test_push_renders_and_pushes` writes into the `-p` dir, so it sees the rendered files before the temp dir is deleted.

- [ ] **Step 4: Run to verify they pass**

Run: `$PY -m pytest tests/test_chain_kaggle.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add deluge/chain/kaggle.py tests/test_chain_kaggle.py
git commit -m "chain: kaggle CLI adapter and kernel rendering

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: One tick, and the initial state file

**Files:**
- Create: `deluge/chain/orchestrate.py`, `.github/chain/state.json`
- Test: `tests/test_chain_orchestrate.py`

**Interfaces:**
- Consumes: everything from Tasks 2, 3 and 5.
- Produces:
  - `tick(kaggle, state_path, queue_path, now, command, commit) -> tuple[int, Optional[str]]`, returning the exit code and the commit message (or None when the state didn't change)
  - `main(argv=None) -> int`, run as `python -m deluge.chain.orchestrate --command tick|resume|skip --message-file PATH`

- [ ] **Step 1: Write the failing tests**

`tests/test_chain_orchestrate.py`:

```python
"""orchestrate.tick against a fake Kaggle: I/O around step(), not step() itself."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from deluge.chain.kaggle import KaggleError, KaggleQuotaError
from deluge.chain.orchestrate import tick
from deluge.chain.step import MAX_API_ERRORS, initial_state

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]
QUEUE = """runs:
  - name: smoke
    model: configs/model/screen.yaml
    train: configs/train/smoke.yaml
    accelerator: cpu
    data: synthetic
"""


class FakeKaggle:
    def __init__(self, status="running", session=None, status_error=None, push_error=None):
        self._status, self._session = status, session
        self.status_error, self.push_error = status_error, push_error
        self.pushes, self.fetches = [], 0

    def status(self, slug):
        if self.status_error:
            raise self.status_error
        return self._status

    def fetch_session_json(self, slug):
        self.fetches += 1
        return self._session

    def push(self, state, action):
        if self.push_error:
            raise self.push_error
        self.pushes.append((state["session"], action))


@pytest.fixture
def files(tmp_path):
    queue = tmp_path / "runs.yaml"
    queue.write_text(QUEUE)
    state = tmp_path / "state.json"
    state.write_text(json.dumps(initial_state(), indent=2) + "\n")
    return state, queue


def started(files, kaggle=None):
    state_path, queue = files
    tick(kaggle or FakeKaggle(), state_path, queue, NOW, "tick", "abc123")
    return json.loads(state_path.read_text())


def test_a_tick_with_nothing_to_do_writes_nothing(files):
    state_path, queue = files
    queue.write_text("runs: []\n")
    before = state_path.read_text()
    code, message = tick(FakeKaggle(), state_path, queue, NOW, "tick", "abc123")
    assert (code, message) == (0, None) and state_path.read_text() == before


def test_idle_tick_pushes_and_records(files):
    kaggle = FakeKaggle()
    state = started(files, kaggle)
    assert state["status"] == "running" and kaggle.pushes[0][1].first is True
    assert kaggle.pushes[0][0] == 1       # pushed with the new session number


def test_a_live_session_is_not_downloaded(files):
    state_path, queue = files
    started(files)
    kaggle = FakeKaggle(status="running")
    tick(kaggle, state_path, queue, NOW + timedelta(hours=1), "tick", "abc123")
    assert kaggle.fetches == 0


def test_api_errors_count_then_fail(files):
    state_path, queue = files
    started(files)
    kaggle = FakeKaggle(status_error=KaggleError("503"))
    codes = [tick(kaggle, state_path, queue, NOW + timedelta(hours=1), "tick", "x")[0]
             for _ in range(MAX_API_ERRORS)]
    assert codes == [0] * (MAX_API_ERRORS - 1) + [1]
    assert json.loads(state_path.read_text())["api_errors"] == MAX_API_ERRORS


def test_quota_refusal_on_push_waits(files):
    state = started(files, FakeKaggle(push_error=KaggleQuotaError("quota")))
    assert state["status"] == "waiting" and state["next_side"] == "a"


def test_a_failed_push_is_not_recorded_as_a_live_session(files):
    state = started(files, FakeKaggle(push_error=KaggleError("boom")))
    assert state["status"] == "idle" and state["api_errors"] == 1


def test_halting_writes_state_before_failing(files):
    state_path, queue = files
    state = started(files)
    later = NOW + timedelta(hours=1)
    for _ in range(2):
        crash = {"run_id": state["run"]["run_id"], "session": state["session"],
                 "exit_code": 1, "tokens_seen": 0}
        code, message = tick(FakeKaggle("complete", crash), state_path, queue,
                             later, "tick", "abc123")
        state = json.loads(state_path.read_text())
        later += timedelta(hours=1)
    assert code == 1 and state["status"] == "halted"
    assert "needs attention" in message


def test_committed_state_file_is_the_initial_state():
    committed = json.loads((ROOT / ".github/chain/state.json").read_text())
    assert committed == initial_state()
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_chain_orchestrate.py -v`
Expected: `ImportError: No module named 'deluge.chain.orchestrate'`.

- [ ] **Step 3: Implement `deluge/chain/orchestrate.py`**

```python
"""One tick of the Kaggle chain: observe, decide, act, record.

Run by .github/workflows/chain.yml every 30 minutes. The decision is
deluge.chain.step's; this module only does the I/O around it. It writes the
state file only when the state changed, so that file's git history reads as
the chain's log, and a failed push is never recorded as a live session.

Exit status 1 means the chain needs a human (halted, or Kaggle unreachable
for ~3 h): the failed workflow run is what emails you. Otherwise 0.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .kaggle import KaggleCLI, KaggleError, KaggleQuotaError
from .queue import load_queue
from .step import (MAX_API_ERRORS, Fail, Observation, Push, api_error, kernel_slug,
                   push_refused, step)

STATE = Path(".github/chain/state.json")
QUEUE = Path("configs/chain/runs.yaml")
# Only a finished kernel has a session.json worth downloading.
FINISHED = ("complete", "error", "cancelled", "unknown")


def observe(kaggle, state: dict) -> Observation:
    slug = kernel_slug(state["run"]["name"], state["side"])
    status = kaggle.status(slug)
    session = kaggle.fetch_session_json(slug) if status in FINISHED else None
    return Observation(status=status, session=session)


def tick(kaggle, state_path: Path, queue_path: Path, now: datetime, command: str,
         commit: Optional[str]) -> Tuple[int, Optional[str]]:
    state = json.loads(state_path.read_text())
    queue = load_queue(queue_path)
    try:
        needs_look = state["status"] == "running" and command == "tick"
        observation = observe(kaggle, state) if needs_look else None
    except KaggleError as error:
        new, action = api_error(state, str(error))
    else:
        new, action = step(state, queue, observation, now, command, commit)
        if isinstance(action, Push):
            try:
                kaggle.push(new, action)
            except KaggleQuotaError:
                new, action = push_refused(new, now), None
            except KaggleError as error:
                # Keep the old state: the push did not happen, and the next
                # tick will make the same decision and try it again.
                new, action = api_error(state, str(error))

    message = None
    if new != state:
        state_path.write_text(json.dumps(new, indent=2) + "\n")
        message = describe(state, new, action)
    if isinstance(action, Fail):
        print(f"[chain] needs attention: {action.reason}", file=sys.stderr)
        return 1, message
    return 0, message


def describe(old: dict, new: dict, action) -> str:
    name = (new["run"] or old["run"] or {}).get("name", "-")
    if isinstance(action, Fail):
        return f"chain: {name} needs attention -- {action.reason}"
    if isinstance(action, Push):
        return f"chain: {name} session {new['session']} pushed on {action.side}"
    if len(new["done"]) > len(old["done"]):
        last = new["done"][-1]
        verb = "skipped" if last.get("skipped") else "done"
        return f"chain: {last['name']} {verb} after {last['sessions']} session(s)"
    if new["status"] == "waiting":
        return f"chain: {name} waiting for GPU quota until {new['wait_until']}"
    if new["api_errors"] > old["api_errors"]:
        return f"chain: Kaggle API error {new['api_errors']}/{MAX_API_ERRORS}"
    return "chain: state updated"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--command", choices=("tick", "resume", "skip"), default="tick")
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA"),
                        help="commit a newly started run is pinned to")
    parser.add_argument("--message-file", type=Path,
                        help="written with the commit message when the state changed")
    args = parser.parse_args(argv)

    owner = os.environ.get("KAGGLE_USERNAME")
    if not owner:
        parser.error("KAGGLE_USERNAME is not set (a repo secret, in the workflow)")
    code, message = tick(KaggleCLI(owner), args.state, args.queue,
                         datetime.now(timezone.utc), args.command, args.commit)
    if message:
        print(message)
        if args.message_file:
            args.message_file.write_text(message + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Create the initial state file**

```bash
mkdir -p .github/chain
$PY -c "import json; from deluge.chain.step import initial_state; \
open('.github/chain/state.json','w').write(json.dumps(initial_state(), indent=2) + '\n')"
```

- [ ] **Step 5: Run to verify they pass**

Run: `$PY -m pytest tests/test_chain_orchestrate.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add deluge/chain/orchestrate.py .github/chain/state.json tests/test_chain_orchestrate.py
git commit -m "chain: one tick -- observe, decide, push, record

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Workflow and docs

**Files:**
- Create: `.github/workflows/chain.yml`
- Modify: `docs/kaggle.md` (append a section), `docs/project-structure.md` (the `deluge/` tree)
- Test: `tests/test_chain_workflow.py`

**Interfaces:**
- Consumes: `python -m deluge.chain.orchestrate` (Task 6).

- [ ] **Step 1: Write the failing test**

`tests/test_chain_workflow.py`:

```python
"""The chain workflow's safety properties, read from the YAML itself."""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/chain.yml"


def load():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML reads the bare key `on` as boolean True.
    return workflow, workflow.get("on", workflow.get(True))


def test_runs_on_a_schedule_and_by_hand_only():
    _, triggers = load()
    assert set(triggers) == {"schedule", "workflow_dispatch"}   # no fork can reach the key
    assert triggers["schedule"] == [{"cron": "*/30 * * * *"}]
    command = triggers["workflow_dispatch"]["inputs"]["command"]
    assert command["options"] == ["tick", "resume", "skip"] and command["default"] == "tick"


def test_ticks_never_overlap_or_cancel_each_other():
    workflow, _ = load()
    assert workflow["concurrency"] == {"group": "chain", "cancel-in-progress": False}


def test_may_commit_the_state_file_and_nothing_else_is_granted():
    workflow, _ = load()
    assert workflow["permissions"] == {"contents": "write"}
    steps = workflow["jobs"]["tick"]["steps"]
    commit = next(s for s in steps if s.get("name") == "Commit state")
    assert "git add .github/chain/state.json" in commit["run"]
    assert "git pull --rebase" in commit["run"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_chain_workflow.py -v`
Expected: `FileNotFoundError` for `chain.yml`.

- [ ] **Step 3: Create `.github/workflows/chain.yml`**

```yaml
# The Kaggle chain's tick. See docs/kaggle.md ("The chain").
#
# Every 30 minutes: read .github/chain/state.json, ask Kaggle about the live
# session, take at most one step, and commit the state file if it changed.
# There is no pull_request trigger on purpose: the Kaggle key must never be
# reachable from a fork's code.
name: chain

on:
  schedule:
    - cron: "*/30 * * * *"
  workflow_dispatch:
    inputs:
      command:
        description: "tick, resume a halted or waiting chain, or skip the current run"
        type: choice
        options: [tick, resume, skip]
        default: tick

concurrency:
  group: chain
  cancel-in-progress: false

permissions:
  contents: write

jobs:
  tick:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e . "kaggle>=1.7"
      - id: tick
        name: Tick
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
          COMMAND: ${{ inputs.command || 'tick' }}
        run: |
          set +e
          python -m deluge.chain.orchestrate --command "$COMMAND" \
            --message-file "$RUNNER_TEMP/message"
          echo "code=$?" >> "$GITHUB_OUTPUT"
      # Commit even when the tick failed: a halt is recorded before it emails.
      - name: Commit state
        if: always()
        run: |
          if [ -s "$RUNNER_TEMP/message" ]; then
            git config user.name "github-actions[bot]"
            git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
            git add .github/chain/state.json
            git commit --quiet -F "$RUNNER_TEMP/message"
            git pull --rebase --quiet
            git push --quiet
          fi
      - name: Fail if the chain needs attention
        if: steps.tick.outputs.code != '0'
        run: exit 1
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_chain_workflow.py -v`
Expected: PASS.

- [ ] **Step 5: Append "The chain" to `docs/kaggle.md`**

````markdown
## The chain

A scheduled GitHub Action works through `configs/chain/runs.yaml`, one
session at a time, with no machine of yours involved. The design is in
`docs/superpowers/specs/2026-09-27-kaggle-chain-design.md`.

**Adding a run.** Append an entry to `configs/chain/runs.yaml` and push to
`master`. The next idle tick (within 30 min) pins the run to that commit
and pushes its first session. Code pushed later affects only later runs.
Names are identities: a name already in `done` never runs again, so rename
to rerun.

**Where things are.**
- Progress: `.github/chain/state.json`, whose `git log` is the chain's history.
- Each session's output (checkpoints, `log.jsonl`, `session.json`) is the
  output of the private kernel `deluge-<run>-a` or `-b`:
  `kaggle kernels output <you>/deluge-<run>-<side> -p out/`.
- Every session's output holds the run's newest checkpoint, so the most
  recent side is all you need.

**When it emails you.** A failed "chain" workflow run means one of two things:
- **Halted:** two failed sessions in a row with no progress. Look at the
  kernel's log on Kaggle, fix the problem, then
  `gh workflow run chain -f command=resume`. To give up on the run instead:
  `-f command=skip`.
- **Kaggle unreachable for ~3 h:** no action is needed if it recovers. The
  chain keeps ticking.

Running out of GPU quota is normal: the chain waits 6 h and retries, and
doesn't email.

**Your local `master` falls behind.** The bot commits state to `master`, so
`git pull --rebase` before pushing.

**Sixty days.** GitHub disables scheduled workflows in a public repo after 60
days without activity. The state commits should count as activity. If the
schedule is disabled anyway, re-enable it under Actions → chain.
````

If Task 1 found one T4 per session, also add under "Budgeting a session": "Pushed sessions get one T4 (docs/kaggle-chain-spike.md), so double the clock times above when the chain runs them."

- [ ] **Step 6: Add the package to `docs/project-structure.md`'s tree**

Under the `deluge/` tree, after the `model/` block, add:

```
│   ├── chain/                  # unattended Kaggle sessions (docs/kaggle.md, "The chain")
│   │   ├── step.py             # the pure decision; every transition is a unit test
│   │   ├── orchestrate.py      # one tick of .github/workflows/chain.yml
│   │   ├── kaggle.py           # kaggle CLI adapter + kernel rendering
│   │   ├── session.py          # what a Kaggle kernel runs
│   │   └── queue.py            # configs/chain/runs.yaml
```

- [ ] **Step 7: Full suite**

Run: `$PY -m pytest -q`
Expected: the baseline count from Task 0 plus every new test, all passing.

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/chain.yml tests/test_chain_workflow.py docs/kaggle.md docs/project-structure.md
git commit -m "chain: the 30-minute tick workflow, and its runbook

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Merge and rehearse on Kaggle CPU

**Needs the user:** permission to merge into `master` and push (this also brings the M1 harness to `master`), and access to the email GitHub sends failure notices to.

**Files:**
- Modify: `configs/chain/runs.yaml` (add, then remove, `smoke-crash`)

- [ ] **Step 1: Ask the user to approve the merge and push.** On a yes:

```bash
cd /home/LAIN/dev/Deluge-chain
git fetch origin
git merge-base --is-ancestor origin/master kaggle-chain && echo fast-forward   # must print; else stop and ask
git push origin kaggle-chain:master
git fetch origin master:master      # local master follows; it is checked out nowhere
```

A fast-forward: `kaggle-chain` contains `master` (`5d1d88b`) plus the M1 harness, so no history is rewritten.

- [ ] **Step 2: Watch the first tick**

The schedule picks it up within 30 minutes. To start sooner:

```bash
gh workflow run chain
gh run list -w chain -L1          # once it is listed:
gh run watch <run id>
git pull origin master && cat .github/chain/state.json
```

Expected: `status: running`, `run.name: smoke`, `session: 1`, `side: a`. The kernel `deluge-smoke-a` appears in the user's Kaggle "Your Work".

- [ ] **Step 3: Drive it through three sessions**

Each smoke session takes about 4 minutes (2 min training plus boot and clone). After each one finishes, and at least 10 minutes after its push (the grace period), run `gh workflow run chain` again. Expected sequence of state commits:
- `smoke session 2 pushed on b`
- `smoke session 3 pushed on a`
- `smoke done after 3 session(s)`

- [ ] **Step 4: Check resume continuity**

```bash
K=kaggle   # the Task 1 scratch venv's kaggle, or pip install kaggle
$K kernels output <owner>/deluge-smoke-a -p $S/smoke-out
$PY - "$S/smoke-out/runs/smoke/log.jsonl" <<'EOF'
import json, pathlib, sys
rows = [json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines()]
starts = [r for r in rows if r.get("event") == "start"]
steps = [r["step"] for r in rows if r.get("event") == "step"]
print("sessions:", len(starts), "resumed_from:", [s["resumed_from"] for s in starts])
print("last step:", steps[-1], "monotonic:", steps == sorted(steps))
EOF
```

Expected: 3 starts. The 2nd and 3rd have a `resumed_from` checkpoint. Steps rise steadily to 500, with no reset to 0.

- [ ] **Step 5: Rehearse the halt**

Append to `configs/chain/runs.yaml`:

```yaml
  # Rehearsal of the halt path: fails before its first step. Remove after.
  - name: smoke-crash
    model: configs/model/screen.yaml
    train: configs/train/smoke.yaml
    model_impl: deluge.train.smoke:build_crashing
    accelerator: cpu
    data: synthetic
```

Commit (`chain: rehearse the halt path`), `git pull --rebase`, push, and drive ticks as in Step 3. Expected:
- `smoke-crash session 1 pushed on a`
- `session 2 pushed on b`
- `needs attention`: the workflow run fails and GitHub emails the user.

Ask the user to confirm the email arrived.

- [ ] **Step 6: Rehearse skip, then clean up the queue**

```bash
gh workflow run chain -f command=skip      # expected commit: "smoke-crash skipped after 2 session(s)"
```

Then remove the `smoke-crash` entry *and* the `smoke` entry from `runs.yaml`, leaving `runs: []` with the header comment. Both are in `done` now. Commit with `chain: rehearsal done; queue empty until M2 lands`, pull, and push. Tell the user the spike and smoke kernels can be deleted from Kaggle (`kaggle kernels delete`, or the website).

- [ ] **Step 7: Remove the worktree**

```bash
cd /home/LAIN/dev/Deluge && git worktree remove ../Deluge-chain
```

Keep the `kaggle-chain` branch until the user says otherwise.
