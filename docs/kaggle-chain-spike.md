# Kaggle chain spike: what the push API actually does

Date: 2026-09-27 · Kaggle CLI 2.2.4 · account `heylaine` · throwaway CPU
kernels `deluge-spike-{a,b,c,d,e,g}`, all deleted afterwards.

The chain design (`docs/superpowers/specs/2026-09-27-kaggle-chain-design.md`)
rests on Kaggle behaviour its docs do not state. Each assumption below was
checked against a real kernel; the constants in `deluge/chain/kaggle.py` and
`session.py` come from here.

| # | Assumption | Verdict |
|---|---|---|
| A1 | Two kernels may list each other in `kernel_sources` | **holds** |
| A2 | A source's output mounts under `/kaggle/input/` | **holds, two layouts** |
| A3 | The mount is the source's latest version | **holds** |
| A4 | `kernels output --file-pattern` fetches only matching files | **holds** (+ the log) |
| A5 | `kernels status` prints `has status "<X>"` | **holds**, enum form |
| A6 | A failed push is detectable | **holds**, exit 1 and no success line |
| A7 | Pushing with a not-yet-existing source | **succeeds and drops the source** |
| A8 | `machine_shape: NvidiaTeslaT4` provisions a T4 | **holds: T4 x2** once phone-verified |

## A1, A3: mutual sources, latest version

`a` v1 (no sources) → `b` v1 (sources `a`) → `a` v2 (sources `b`) → `b` v2
(sources `a`). Both mutual pushes were accepted without warning, and each
session saw the other's newest marker:

```
[a v2] SAW /kaggle/input/deluge-spike-b/runs/spike/marker.txt b1
[b v2] SAW /kaggle/input/notebooks/heylaine/deluge-spike-a/runs/spike/marker.txt a2
```

## A2: the mount path is not stable

Within one hour Kaggle mounted a source both as `/kaggle/input/<slug>/` and as
`/kaggle/input/notebooks/<owner>/<slug>/`. `session.find_prior_run_dir`
therefore globs `**/runs/<name>` rather than building a path, and its test
covers both layouts. The mount also carries Kaggle's own files
(`__script__.py`, `__results__.html`, `__output__.json`, `custom.css`) beside
the kernel's output.

## A4: fetching one file

```
kaggle kernels output heylaine/deluge-spike-a -p o1 --file-pattern '^session\.json$'
  Output file downloaded to o1/session.json
  Kernel log downloaded to o1/deluge-spike-a.log
```

`big.bin` (10 MB) was skipped. The kernel log always comes too; it is small.
The log is a JSON array of `{stream_name, time, data}` records.

## A5: status strings

Seen: `KernelWorkerStatus.QUEUED`, `.RUNNING`, `.COMPLETE`, `.ERROR`, printed
as `heylaine/<slug> has status "KernelWorkerStatus.COMPLETE"`. Immediately
after a push the status was already the new version's (`RUNNING`), not the
previous version's `COMPLETE` — but the chain keeps its 10-minute grace
anyway, since one observation is not a guarantee. Cancel states were not
observed; the adapter maps `CANCEL_REQUESTED`/`cancelAcknowledged` forms.

## A6: failed pushes

Invalid metadata (`"language": "cobol"`): exit code **1**, stdout
`A valid language must be specified in the metadata. ...`, no
`successfully pushed`. Successful pushes print
`Kernel version N successfully pushed.  Please check progress at ...`.
The adapter requires that line, so either signal is caught.

## A7: unresolvable sources are dropped silently

```
The following are not valid kernel sources and could not be added to the kernel: ['heylaine/deluge-spike-nope']
Kernel version 1 successfully pushed.
```

Exit 0; the kernel runs without the source. Consequence for the chain: a
handed-off session that finds no prior output must **refuse to train**, or it
would restart the run from step 0. `session.main` raises in that case (the
push's `first` flag says whether prior output is expected).

## Errored kernels keep their output

`deluge-spike-c` wrote its files then raised: status `ERROR`, and
`kernels output` still returned `session.json`, `runs/`, and `big.bin`. The
chain does not rely on this (session.py always exits 0), but it means a
Kaggle-level error after training still leaves the checkpoint recoverable.

## A8: T4 x2, but only on a phone-verified account

First attempt: `deluge-spike-g` pushed with `enable_gpu: true, machine_shape:
NvidiaTeslaT4`; `kernels pull -m` showed both recorded, yet the session had no
`nvidia-smi` and ended `ERROR` on the script's own `FileNotFoundError`. No
warning from the push. The account was not phone-verified.

After phone verification, the same push:

```
GPU 0: Tesla T4 (UUID: GPU-a7a7e281-...)
GPU 1: Tesla T4 (UUID: GPU-857ea636-...)
CUDA 2 [(7, 5), (7, 5)]
```

So the CLI's `NvidiaTeslaT4` shape is the editor's "GPU T4 x2", and the
`docs/kaggle.md` budgets (written for 2xT4) stand. Verification also gates
internet: the chain's first smoke session, pushed minutes before it, could not
clone the repo, reported exit 4, and was retried on the same side.

Consequence for the chain: a GPU run on an account that cannot get a GPU
would report "no GPU" (exit 3) every session and wait forever. The chain
therefore fails once — emailing — after a week of consecutive no-GPU waits,
since quota resets weekly and a longer drought is not quota.

## Environment facts

- Kaggle runs Python 3.12 (`/usr/local/lib/python3.12/dist-packages`).
- `/kaggle` and `/tmp` are writable.
- The Kaggle username (`heylaine`) differs from the GitHub one (`heylain`);
  the chain takes the owner from `KAGGLE_USERNAME` and the repo URL from
  `kaggle.REPO`, so they need not match.
