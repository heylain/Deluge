# M1: brainstorm notes (in progress, not yet a spec)

Date: 2026-10-01 · Status: brainstorming. Two decisions made, one question open.
The design spec gets written once the design is agreed; these notes are where a
fresh session picks up.

## Decided

**1. M1 is the spec's M1, end to end** (`cfc-hybrid-spec.md` §13): build the
model, get A0 training stably against a matched tiny transformer, run the A1
arms through the Kaggle chain, and make the go/no-go call. If CfC is not better
than the gated conv within noise, text work freezes and the architecture
retargets to audio/event tokens (§11).

**2. Order: A0 before the Triton scan kernel.**

1. Model with a **head-batched chunked scan in plain PyTorch**.
2. **Throughput probe** on the chain: a short timed run of the real dev model on
   T4 x2.
3. A0 stability run.
4. A1 arms. A Triton scan is added first only if step 2 requires it.

Why:
- A0 answers the riskiest question in the project: does the scan train stably
  at all. A fast kernel for a recurrence that does not train is wasted work.
- A kernel has no speed target until something trains.
- λ is per-head (§3), so within a chunk the decay matrix is `[B, H, C, C]` and
  the scan is a batched matmul against `bv` reshaped to `[B, H, C, d_inner/H]`.
  The reference `chunked_scan` materialises `[B, C, C, d_inner]`, roughly 100x
  more. A0 and A1 run at Δ = 1 everywhere (no MoD until A3), so the simple form
  covers both. It must pass the existing parity tests against `sequential_scan`.
- `docs/project-structure.md`'s commit order already puts Triton after the
  model, A0 and train/infer equivalence.

**The guard.** `configs/train/screen.yaml`'s "~6 h per arm" assumes ~23k tok/s
across 2xT4, and nobody has measured it. If the probe comes in far below that,
the Triton scan moves ahead of the A1 runs. That is a measurement, not a
judgement call. If it is fast enough, Triton is deferred to M3, where paid H100
time makes speed matter.

## Open: what counts as "within noise" for the go/no-go

The CfC arm doubles as A0, so the minimum is four runs at the 500M-token
screening budget: CfC, gated conv (B3), LRU (B4), and the matched tiny
transformer. That's ~24 h of the 30 h/week free quota, at the unmeasured ~6 h/arm.

- **(a) Recommended:** one seed per arm, plus a second CfC seed. The CfC
  seed-to-seed gap is the noise floor. ~30 h, about one week.
- (b) Two seeds of every arm. ~48 h, about two weeks; a per-arm noise estimate.
- (c) One seed each and a fixed margin chosen now (e.g. CfC beats conv by
  >= 0.02 nats held-out). ~24 h, but the margin is a guess.

## Housekeeping found along the way

- **Milestone numbers drift.** The spec's M1 is "A1 result, go/no-go"; the
  harness branch and commit called the harness M1; the chain spec calls
  `deluge.model:build` M2. The spec's M0 also includes the scan kernel and A0,
  neither of which exists yet. Settle one numbering as part of the M1 spec.
- **State at this point.** Done: config with derived param counts, training
  harness, Kaggle chain (rehearsed, queue `runs: []`). Committed on branch `m1`:
  reference `sequential_scan`/`chunked_scan` with parity tests, `MixerState`,
  `RMSNorm`, `a_min`/`a_max` decay-range config. Missing: CfC / gated-conv / LRU
  mixers, window attention, dense FFN, block, model, `deluge.model:build`, the
  tiny-transformer baseline.

## Constraints for whoever picks this up

- **Compute is free tier only.** Kaggle T4 x2 (sm_75, fp16, 30 GPU-h/week,
  12 h sessions). Never P100: sm_60, and Triton needs sm_70+. A local GPU is for
  short debug runs only, not training.
- **Training runs unattended** through the chain (`docs/kaggle.md`, "The chain"):
  add a run to `configs/chain/runs.yaml` on `master` and push. Runs pin a
  commit, so model code must be on `master` before a real arm is queued.
- **Real arms also need** the tokenized dataset uploaded to Kaggle once
  (`docs/kaggle.md`).
