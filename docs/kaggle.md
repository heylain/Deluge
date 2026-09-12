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
    --model configs/model/dev.yaml \
    --train configs/train/screen.yaml \
    --data /kaggle/input/deluge-tokens/tokens.bin \
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
| 0 | budget spent, the run is done |
| 2 | out of time, resumable — queue another session |

## Data

Tokenize once, upload as a Kaggle Dataset, attach it read-only to every arm:

```bash
python scripts/prepare_data.py \
    --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --tokens 600_000_000 --out data/tokens.bin
```

600M tokens (1.2 GB) covers a screening arm with margin. A dev arm needs 2.26B,
which is 4.5 GB — under Kaggle's dataset limit, and worth preparing once since
every arm must see identical data for the ladder to mean anything.

## Budgeting a session

| | tokens | ~wall clock on 2xT4 | sessions |
|---|---|---|---|
| screening arm (`train/screen.yaml`) | 500M | ~6 h | 1 |
| dev arm (`train/dev.yaml`) | 2.26B | ~27 h | 3 |

These assume ~16 TFLOP/s delivered across the pair, which is a guess until the
first run measures it. `log.jsonl` records `tokens_per_second` every step —
take the number from there and re-derive before planning a ladder around it.

## Checkpoint size

The checkpoint holds fp32 weights, both Adam moments and the scaler. For the
113M dev config that is roughly 1.8 GB, so `keep_last: 2` is ~3.6 GB of the
20 GB working directory. Drop to `keep_last: 1` if that gets tight — rotation
never deletes the only checkpoint.

## If a session dies badly

`load_latest` tries checkpoints newest-first and skips any that fail to
deserialize, so a kill during a write costs one cadence interval, not the run.
Writes are atomic (temp file, fsync, rename), so a half-written checkpoint never
appears under a real name in the first place.
