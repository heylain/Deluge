# Running on Kaggle

Free tier: **30 GPU-hours per week, 12 hours per session.** A screening arm fits
in one session; a dev arm takes three. The harness exists so that the second
fact costs nothing but wall-clock.

## Pick T4 x2, never P100

Accelerator → **GPU T4 x2**.

The P100 is compute capability 6.0, and Triton requires 7.0 or newer. The
chunked parallel scan of spec 3 will not compile on it at all — this is not a
speed difference, it is a hard stop. The T4 is 7.5 and works.

The T4 has fp16 tensor cores but no bf16 ones, which is why
`configs/train/base.yaml` sets `dtype: fp16`. On anything sm_80 or newer
(A100, 4090, 5070 Ti, H100) set `dtype: bf16` — but note that dtype is part of
the resume fingerprint, so switching it starts a new run rather than continuing
an old one.

## Session shape

```python
!git clone https://github.com/<you>/Deluge /kaggle/working/Deluge
%cd /kaggle/working/Deluge
!pip install -qe .

!python -m deluge.train \
    --model configs/ablations/A0_scan_only.yaml \
    --train configs/train/screen.yaml \
    --data /kaggle/input/deluge-long4k/train.bin \
    --out /kaggle/working/runs/A0 \
    --resume-from /kaggle/input/deluge-A0-prev \
    --data-parallel
```

Then **Save Version**. `/kaggle/working` becomes a dataset; attach it as
`--resume-from` on the next session and the run continues from the step it
stopped at.

`--data-parallel` splits each micro-batch across both T4s. It is the inefficient
way to use two GPUs (~1.6x against DDP's ~1.9x) but it works inside a notebook,
where `torchrun` does not. Checkpoints are written unwrapped, so a two-GPU
session resumes on one GPU unchanged.

## What the deadline does

`deadline_minutes: 660` stops the run at 11 h, an hour before Kaggle's kill, and
checkpoints. The loop stops *before* a step it predicts will overrun rather
than losing a partial one, and it also handles SIGTERM, so an early kill still
lands on a checkpoint.

Exit codes:

| code | meaning |
|---|---|
| 0 | the run is done: budget spent and, with a held-out set, `final_eval.json` written |
| 2 | out of time, resumable — queue another session |

With a `heldout.bin` beside the training data, the run evaluates on it every
`eval_every_tokens` and once more, on the whole set, when the budget is spent.
A session killed during that final eval leaves the run resumable, and the next
session runs only the eval.

## Data

Every M1 arm trains at `seq_len` 4096 on long documents, and every arm
attaches the same private dataset, `heylaine/deluge-long4k`:

| file | what |
|---|---|
| `train.bin` | ~2.4B tokens: documents of at least 4,097 Mistral tokens, cut into whole 4,097-token pieces (the dev budget's 2.26B plus margin) |
| `heldout.bin` | ~24M tokens: the 1% of documents whose text hashes there, in hash order |
| `provenance.json` | source and revision, tokenizer, `piece_len`, pre-filter, split rule, counts |

A piece is 4,097 tokens because a sequence reads 4,096 inputs plus one shifted
target, and a piece never spans two documents. `deluge.train` reads the data in
pieces whenever `provenance.json` names a `piece_len`, and refuses to start if
it is not `seq_len + 1`. It picks up `heldout.bin` and `provenance.json` from
beside `--data`, so the chain's `data:` field stays
`heylaine/deluge-long4k:train.bin`.

How the dataset is made:

1. **Measure** the yield before fixing a size. This reads only FineWeb-edu's
   `token_count` column (GPT-2 tokens) — on the 10BT sample about 50 MiB, in
   about 10,000 small range requests — plus a few thousand documents to
   calibrate the GPT-2 → Mistral ratio and suggest a pre-filter. Install the
   versions the build kernel pins (`PIP_PINS` in `scripts/kaggle_prep.py`), so
   the pre-filter is calibrated with the tokenizer that builds the data:

   ```bash
   pip install huggingface_hub==1.33.0 pyarrow==25.0.1 tokenizers==0.23.2
   python scripts/prepare_data.py measure \
       --dataset HuggingFaceFW/fineweb-edu --files 'sample/10BT/*.parquet' \
       --tokenizer mistralai/Mistral-7B-v0.1
   ```

   If `yield.tokens` is under ~2.43B (2.4B to train.bin, plus the 1% held
   out), use `--files 'sample/100BT/*.parquet'`: the long mode stops once
   train.bin is full, so the larger sample costs only download time.
2. **Build** it in a Kaggle CPU kernel with internet, so the download and
   tokenizing never run on the home PC. `scripts/kaggle_prep.py` renders the
   kernel, which clones this repo at a pinned commit (it must be on GitHub)
   and runs `prepare_data.py long` into `/kaggle/working`:

   ```bash
   python scripts/kaggle_prep.py --owner heylaine --commit <sha> --dir /tmp/prep-kernel -- \
       --dataset HuggingFaceFW/fineweb-edu --files 'sample/10BT/*.parquet' \
       --revision <sha that measure printed> --tokenizer mistralai/Mistral-7B-v0.1 \
       --prefilter <from measure> --train-tokens 2_400_000_000
   KAGGLE_CONFIG_DIR=~/Downloads kaggle kernels push -p /tmp/prep-kernel
   ```
3. **Publish** the kernel's output as the private dataset `deluge-long4k`: on
   Kaggle, open the kernel's output and create a new dataset from it. If that
   route is not available, download the output and upload it with
   `kaggle datasets create`.

The pre-filter, the source revision and the split rule are in
`provenance.json`, which the trainer copies into every run directory and the
exporter puts in each run's Hugging Face `run.json`.

## Budgeting a session

| | tokens | ~wall clock on 2xT4 | sessions |
|---|---|---|---|
| screening arm (`train/screen.yaml`) | 500M | ~6 h | 1 |
| dev arm (`train/dev.yaml`) | 2.26B | ~27 h | 3 |

These assume ~16 TFLOP/s delivered across the pair, which is a guess until the
first run measures it, and were made at `seq_len` 1024. At 4096 the hybrid
costs roughly 17% more and the all-attention transformer roughly 70% more.
`log.jsonl` records `tokens_per_second` every step: re-derive this table from
the first A0 session's number before planning around it.

## Memory

`micro_batch` in `configs/train/base.yaml` is split across the two T4s by
`--data-parallel`, so it is twice the per-GPU batch. It is the largest power of
two (so it divides a step's 64 sequences) whose peak *reserved* memory per GPU
stays at or under ~13 GiB, on the largest M1 arm. A T4 has ~14.7 GiB usable;
13 leaves margin, and on the 5070 Ti, where it is measured, the display uses
some too. `log.jsonl` records `peak_alloc_gib` and `peak_reserved_gib` every
step, the max over the GPUs in use.

If even one sequence per GPU does not fit, the first lever is the loss: the
fp32 cross-entropy over the 32k vocab is ~1–1.5 GB per 4,096-token sequence,
and computing it in sequence chunks recovers most of that. Per-block activation
checkpointing comes after.

## Checkpoint size

The checkpoint holds fp32 weights, both Adam moments and the scaler. For the
110M M1 arms that is ~1.3 GB, so `keep_last: 2` is ~2.6 GB of the 20 GB
working directory. Drop to `keep_last: 1` if that gets tight — rotation never
deletes the only checkpoint.

## If a session dies badly

`load_latest` tries checkpoints newest-first and skips any that fail to
deserialize, so a kill during a write costs one cadence interval, not the run.
Writes are atomic (temp file, fsync, rename), so a half-written checkpoint never
appears under a real name in the first place.

## The chain

A scheduled GitHub Action works through `configs/chain/runs.yaml`, one
session at a time, with no machine of yours involved. The design is in
`docs/superpowers/specs/2026-09-27-kaggle-chain-design.md`.

**Adding a run.** Append an entry to `configs/chain/runs.yaml` and push to
`master`. M1's runs set `export: true` (see "Results and checkpoints"). The next idle tick (within 30 min) pins the run to that commit
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

- **No GPU for a week:** every session reported no usable GPU for 7 days,
  which is longer than the quota cycle. Check that the Kaggle account is
  phone-verified (Kaggle requires it for GPU, and a GPU kernel on an
  unverified account silently runs without one: `docs/kaggle-chain-spike.md`).
  The chain keeps retrying.

Running out of GPU quota is normal: the chain waits 6 h and retries, and
doesn't email.

**Your local `master` falls behind.** The bot commits state to `master`, so
`git pull --rebase` before pushing.

**Sixty days.** GitHub disables scheduled workflows in a public repo after 60
days without activity. The state commits should count as activity. If the
schedule is disabled anyway, re-enable it under Actions → chain.

## Results and checkpoints

`.github/workflows/export.yml` runs hourly, apart from the chain's tick. For
each done run with `export: true` it takes two independent steps, each skipped
once its marker is in the repo:

1. **Results.** It fetches `final_eval.json` and `log.jsonl` from the kernel
   that ran the run's last session, and commits `final_eval.json`,
   `evals.csv` and a downsampled `train.csv` to `docs/results/runs/<name>/`.
   `scripts/m1_verdict.py` reads only these.
2. **Checkpoint.** It uploads the newest `step-*.pt`, the full `log.jsonl` and
   a `run.json` (the queue entry, the done record, the data provenance) to
   `<name>/` in the private Hugging Face repo named in
   `configs/chain/export.yaml`, then commits `docs/results/runs/<name>/export.json`
   (repo, path, sha256, size).

Setup, once, before the first exported run:
- Create a Hugging Face token that can write to and create repos in your
  namespace, and add it as the repo secret `HF_TOKEN`
  (`gh secret set HF_TOKEN`).
- Set `hf_repo: <hf-user>/deluge-runs` in `configs/chain/export.yaml`. The
  repo is created private if it does not exist.

A step that fails writes nothing, and the next hourly run retries it. The
workflow fails, and so emails you, only for a run done more than 3 h without
that step succeeding; then it fails every hour until fixed. A Hugging Face
failure never holds up the results.

Each checkpoint is ~1.3 GB; M1's twelve exported runs come to ~16 GB of the
free account's 100 GB of private storage. If the export cannot be fixed in
time, the checkpoint can still be fetched by hand while Kaggle keeps the
kernel's output:
`kaggle kernels output heylaine/deluge-<run>-<side> -p out/`.
