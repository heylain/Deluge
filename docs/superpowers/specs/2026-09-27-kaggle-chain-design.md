# Kaggle chain: unattended training across 12 h sessions

Date: 2026-09-27 · Status: design approved in chat, awaiting spec review

## Goal

Deluge trains in the background on Kaggle's free tier, with no local machine
involved. A queue of runs is worked through one 11 h session at a time; each
session resumes from the last one's checkpoint. The user edits a queue file,
pushes, and comes back to finished runs.

**Success:** add runs to `configs/chain/runs.yaml`, push to `master`, touch
nothing for days, find each run's complete `log.jsonl` and final checkpoint in
its Kaggle kernel output, and the chain's history in `git log` of the state
file.

## Constraints

- **$0.** Kaggle free tier (T4 x2, 30 GPU-h/week, 12 h sessions); GitHub
  Actions on a public repo (`heylain/Deluge`, default branch `master`).
- **The user's PC is not involved.** It is needed for uni and is not left on.
- **A run's code is fixed for its lifetime.** The resume fingerprint means a
  run cannot change code mid-flight; the chain pins each run to one commit.
- **Quota is precious.** A broken commit must not burn sessions retrying.

## Non-goals

One run at a time only (no parallel runs). No Modal or other backend. No
collection of results into `docs/results/`. No quota accounting beyond
"detect and wait". No notifications beyond GitHub's failed-workflow email.

## Prerequisites

- `m1-training-harness` merged into `master`. `master` (`5d1d88b`) predates the
  trainer; scheduled workflows run only from the default branch, and sessions
  clone the pinned commit, which must contain `deluge.train`.
- Repo secrets `KAGGLE_USERNAME`, `KAGGLE_KEY` (done, 2026-09-27).
- Kaggle account phone-verified (GPU and internet in kernels need it).
- For real arms: the tokenized dataset uploaded once (`docs/kaggle.md`), and
  M2's `deluge.model:build`. Until then the queue holds only `smoke`.

## Handoff mechanism: two kernels per run

Each run owns two private Kaggle script kernels, `deluge-<run>-a` and
`deluge-<run>-b`. Sessions alternate between them. Each kernel lists the
other in `kernel_sources`, so session n+1 mounts session n's output read-only
at `/kaggle/input/deluge-<run>-<other>/`. No checkpoint passes through GitHub.

The first session of a run is pushed on side `a` with **no** `kernel_sources`
(side `b` does not exist yet, and a same-named kernel from an earlier run of
that name must not leak in).

**Invariant: every session's output contains the run's newest checkpoint.**
`session.py` copies the newest loadable checkpoint (and the prior
`log.jsonl`) from its input into its own output before training, so the
output is self-contained even if the session dies before its first save.

## Components

All chain code lives in `deluge/chain/`. Only `session.py` imports torch, and
only after the package is installed inside the kernel; the orchestrator runs
in Actions with the base install (pyyaml, numpy) plus the `kaggle` CLI.

### `configs/chain/runs.yaml` — the queue

```yaml
runs:
  - name: smoke                        # [a-z0-9-]+, becomes the kernel slug
    model: configs/model/screen.yaml
    train: configs/train/smoke.yaml
    model_impl: deluge.train.smoke:build   # optional; default deluge.model:build
    accelerator: cpu                   # cpu | gpu (gpu means T4 x2)
    data: synthetic                    # synthetic | <owner>/<dataset>:<file>
```

Ordered. The next run is the first entry not in `state.done`.

### `.github/chain/state.json` — where the chain is

```json
{
  "status": "idle | running | waiting | halted",
  "run":    { "...": "snapshot of the runs.yaml entry",
              "commit": "<sha pinned at run start>",
              "run_id": "<name>@<start iso time>" },
  "side": "a", "session": 3, "pushed_at": "<iso>", "handed_off": true,
  "last_tokens": 0, "crash_streak": 0, "api_errors": 0,
  "wait_until": null, "next_side": null,
  "done": [ { "name": "smoke", "run_id": "...", "commit": "...",
              "sessions": 3, "finished": "<iso>" } ]
}
```

`session` counts pushes within the run, retries included, so each push has
an identity (`run_id`, `session`) no earlier push shares. `handed_off` turns
true once any session of the run has written a fresh `session.json` (so some
output exists to source); until then pushes carry no `kernel_sources`.
`next_side` is the side a `waiting` or `halted` chain will push when it moves.

The run entry is snapshotted at start, so editing `runs.yaml` mid-run affects
only later runs. The file is committed by the workflow **only when it
changes**, so its git history is the chain's log.

### `deluge/chain/session.py` — what a kernel runs

Pushed as the kernel's code file, with the session's parameters rendered into
a `RUN = {...}` block at the top (run entry, commit, `run_id`, session number,
side, the other side's slug or none). Steps:

1. If `accelerator: gpu` and there is no CUDA device of compute capability
   7.0 or newer: write `session.json` with exit code 3 and stop. (Quota
   exhausted, no GPU, or a P100, which is sm_60 and cannot run Triton. This
   runs before anything else, so it costs about a minute.) The check is on
   capability, not on "two GPUs": the push API may provision one T4 or two
   (see Known risks).
2. Clone `https://github.com/heylain/Deluge` and check out the pinned commit;
   `pip install -e .`.
3. Data: `synthetic` → generate a small uint16 token file in the session;
   otherwise use the attached dataset's file.
4. Carry forward: from `/kaggle/input/<other>/runs/<name>/`, copy the newest
   checkpoint that deserializes (using `Checkpointer.candidates` and the same
   loader training uses, so the rule cannot drift) and `log.jsonl` into
   `/kaggle/working/runs/<name>/`.
5. Run `python -m deluge.train --model … --train … --data … --out
   /kaggle/working/runs/<name> --model-impl …` (plus `--data-parallel` when two or more GPUs are visible,
   `--device cpu` on cpu).
6. Always (in `finally`) write `/kaggle/working/session.json`:
   `{run_id, session, side, exit_code, tokens_seen, commit}`; `tokens_seen`
   from the last `log.jsonl` line. Exit 0 so Kaggle keeps the output.

### `deluge/chain/step.py` — the decision

A pure function `step(state, queue, observation, now, command) -> (state,
action)`. No I/O. `observation` is `{status, session_json}` for the live
kernel, where `status ∈ {queued, running, complete, error, cancelled,
unknown}` and `session_json` is `None` if absent **or stale** (its `run_id`
or `session` does not match the state). `action` is one of `none`,
`push(side, first: bool)`, `fail(reason)`.

### `deluge/chain/kaggle.py` — the adapter

Thin wrapper over the `kaggle` CLI: `status(slug)`, `fetch_session_json(slug)`
(fetches only that file), `push(side, rendered_dir)`. Renders
`kernel-metadata.json`: `is_private: true`, `enable_internet: true`,
`enable_gpu` and `machine_shape: NvidiaTeslaT4` from `accelerator`, `dataset_sources` from `data`,
`kernel_sources` = `[other]` unless `first`. Every call raises one
`KaggleError` type on failure.

### `deluge/chain/orchestrate.py` and `.github/workflows/chain.yml`

`python -m deluge.chain.orchestrate [--command tick|resume|skip]`: load
state and queue, observe via the adapter, call `step`, carry out the action,
write state. The workflow:

- triggers: `schedule: */30 * * * *` and `workflow_dispatch` with a
  `command` input (`tick` default, `resume`, `skip`). No `pull_request`
  trigger, so fork PRs never see the secrets.
- `concurrency: {group: chain, cancel-in-progress: false}`.
- `permissions: contents: write`; commits `.github/chain/state.json` if it
  changed, message `chain: <transition>`, then pushes.
- The pinned commit for a new run is the `GITHUB_SHA` of the tick that
  starts it.

## Transitions

Normal path:

| State / observation | Next |
|---|---|
| idle, queue has a run not in `done` | snapshot it, pin commit, `push(a, first)`, → running, session 1 |
| idle, nothing queued | nothing |
| running, less than 10 min since the push | nothing (the status read may still be the previous version's) |
| running, `queued`/`running` | nothing |
| running, `complete`, exit 2 with progress | `push(other side)`, update `last_tokens` |
| running, `complete`, exit 0 | append to `done`, → idle (the next tick starts the next run) |

Failures:

| Case | Detection | Action |
|---|---|---|
| Training crash | `session_json.exit_code ∉ {0, 2, 3}` | `push(other side)` (output holds the carried checkpoint); `crash_streak` += 1 |
| Kaggle-level kill (OOM, clone/pip failure, hard timeout) | `error`/`cancelled`, or `complete` with no fresh `session_json` | `push(same side)` — its input is unchanged and holds the last good checkpoint; `crash_streak` += 1 |
| Stuck | `queued`/`running` and `now - pushed_at > 13 h` | treated as a Kaggle-level kill |
| Progress | `tokens_seen > last_tokens` | `crash_streak` = 0 |
| Exit 2 without progress | resumable, but `tokens_seen` did not grow | counted as a training crash: a session that trains nothing must not loop forever |
| Repeated failure | `crash_streak` reaches 2 | → halted; `fail` once (the workflow run fails, GitHub emails) |
| Halted | status halted, command `tick` | nothing, exit 0 (no repeat emails) |
| No GPU / quota | exit code 3, or `push` raises a quota error | → waiting, `wait_until = now + 6 h`; not a crash |
| Waiting | `now >= wait_until` | `push(same side)`, → running |
| API/network error | adapter raises `KaggleError` elsewhere | state unchanged except `api_errors` += 1; `fail` when it reaches 6 (~3 h); reset on any successful observation |
| `resume` command | status halted or waiting | `crash_streak` = 0, `push(same side)`, → running |
| `skip` command | any status with a current run | append the run to `done` with `skipped: true` (so it is not picked again), → idle |

Every push increments `session`, and is `first` exactly when `handed_off`
is false, so a retried first session still has no `kernel_sources`.

## Testing

1. **Spike (throwaway, before any chain code).** Two tiny CPU kernels that
   reference each other. Confirm: mutual `kernel_sources` is accepted; the
   mount path is `/kaggle/input/<slug>/`; it mounts the source's **latest**
   version; `kaggle kernels output` can fetch `session.json` alone; the
   exact status strings; whether an errored kernel keeps its output; whether
   a push may reference a not-yet-existing kernel; how the push CLI reports
   an error (exit code or only stdout); and, in one GPU kernel of a few
   minutes, what `machine_shape: NvidiaTeslaT4` provisions (`nvidia-smi -L`:
   one T4 or two). Any mismatch comes back to
   the user as a design change before implementation. Spike kernels are
   deleted afterwards.
2. **Unit tests** (`tests/test_chain_*.py`, no network, no GPU, no torch
   except where `session.py`'s carry-forward is tested against real
   `Checkpointer` files):
   - `step`: one test per row of both transition tables.
   - orchestrate: a tick with nothing to do writes nothing; halting commits
     state before failing; stale `session_json` is ignored.
   - `session.py`: carry-forward picks the newest *loadable* checkpoint and
     skips a corrupt one; `session.json` is written when training raises;
     the GPU check writes exit 3.
   - The existing suite stays green.
3. **Rehearsal on Kaggle CPU** (no GPU quota): `smoke` run with
   `deluge/train/smoke.py` (a tiny embedding → linear LM, labelled test-only)
   and `configs/train/smoke.yaml` (a budget that needs three sessions under
   its deadline). Ticks triggered by hand via `workflow_dispatch`. Passes if
   the chain goes a → b → a, session 3's `log.jsonl` continues from session
   2's last step, and the state lands in `done`. Then one forced crash
   (a smoke config that raises) to watch it halt and email.

## Known risks

- **Kaggle behaviour the spike has not yet confirmed** — above, item 1.
- **GitHub disables schedules after 60 days without repo activity** on public
  repos. State commits should count; to verify. If not, it is one click to
  re-enable.
- **T4 x2 through the push API is undocumented.** The editor offers
  "GPU T4 x2"; the CLI documents only `machine_shape: NvidiaTeslaT4`. If it
  provisions one T4, sessions run single-GPU (about half the throughput in
  `docs/kaggle.md`'s table) and every budget there doubles in sessions; the
  chain itself is unaffected.
- **Kaggle API credential format.** Designed on the legacy username + key;
  if Kaggle retires it, only the adapter's auth changes.
