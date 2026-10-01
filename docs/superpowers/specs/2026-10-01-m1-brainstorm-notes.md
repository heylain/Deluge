# M1: brainstorm notes (in progress, not yet a spec)

Date: 2026-10-01 · Status: brainstorming. Open questions settled; design next.

**Handoff.** The M1 design spec and implementation plan are to be written in a
Claude cloud session (Fable), working on branch `m1`: resume brainstorming from
these notes, settle the open questions with the user, write the spec to
`docs/superpowers/specs/`, then the plan to `docs/superpowers/plans/`.
Implementation then happens locally against that plan.

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

**3. The A0 baseline transformer is our own `Block` stack, all-attention**
(decided 2026-10-01). A new `mixer.type: attention` fills every slot, and the
model is matched to A0's 110.0M params by config, the way `A1_conv_baseline.yaml`
is matched. Full causal attention is just `window` = `seq_len`.

Why not a separate nanoGPT:
- It would still share the harness, data, tokenizer and schedule, or the arms
  would not be matched. So only the forward pass would be independent code.
- The real shared-code risk is in attention. The baseline is 100% attention and
  the hybrid 25%, so a degraded attention layer hurts the baseline more and
  flatters CfC. A brute-force reference test of the attention block (naive
  softmax(QK^T)V against ours, the `sequential_scan` vs `chunked_scan` pattern)
  covers the forward pass without a second model.
- This baseline is B1 at M3 (§10, §13). As a config, B1 at target scale costs
  nothing more.

**4. The noise rule is adaptive with a floor** (decided 2026-10-01). It is
fixed now, before any result exists, so it cannot be fitted to the outcome.
The metric is held-out loss in nats at the end of the 500M-token screening
budget.

Week 1, ~30 h at the unmeasured ~6 h/arm, queued in this order:

1. CfC, seed 1234. This run is A0.
2. The all-attention transformer, seed 1234.
3. CfC, a second seed.
4. Gated conv (B3), seed 1234.
5. LRU (B4), seed 1234.

`seed` sets both init and data order (`TokenStream` takes `cfg.seed`), so
every seed-1234 arm sees identical batches. That makes the comparison paired at
no cost. The second CfC seed changes both, which overstates the noise slightly:
the bar gets stricter, not looser.

- **noise** = max(|CfC seed 1 − CfC seed 2|, 0.01 nats). This estimates the
  standard error of a difference between two single runs. The floor stops a
  lucky close pair from setting the bar too low.
- **A0 passes** if training is stable and CfC is within 1× noise of the
  transformer, or better. Otherwise stop and debug before running A1.
- **A1 verdict.** gap = conv − CfC, both on seed 1234; positive means CfC is
  better.
  - gap ≥ 2× noise: **go**.
  - gap < 1× noise: **no-go**. Pivot per §11.
  - In between: run one more conv seed (paired with CfC seed 2's seed) and a
    third CfC seed, ~12 h. Then decide once, on means, with no further runs.
    Re-estimate noise = max(√2 · sample std of the three CfC seeds, 0.01). With
    two seeds this is exactly |s1 − s2|, so it is the same quantity. **Go** iff
    mean(conv) − mean(CfC) ≥ 1.3× noise. That is the week-1 bar of two standard
    errors, rescaled for 3-vs-2 means: 2·√(1/3 + 1/2)/√2 ≈ 1.29.
- LRU is not part of the go/no-go. It reports what the Δ argument and the
  short conv are each worth.

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
- **The A1 arms are on the wrong rung.** `A1_conv_baseline.yaml` and
  `A1_lru_baseline.yaml` extend `dev.yaml`, which has chunk summaries on.
  A0 has them off, and the ladder (§11) adds them at A2. If the CfC arm doubles
  as A0, both baselines must extend `A0_scan_only.yaml`. Matched against A0's
  110.0M: conv `d_inner` 928 → 872 (+0.08%, a multiple of 8 for fp16 tensor
  cores); LRU stays 1152 (−0.07%). Consequence: M1 needs no chunk-summary
  attention at all, which is M2 work in §13 anyway.
- **The harness has no held-out eval.** The dataset is 600M tokens and an arm
  uses 500M, but `TokenStream` permutes over the whole file. M1 needs a
  held-out slice the stream never draws from, and an eval pass at checkpoints.

## Constraints for whoever picks this up

- **Compute is free tier only.** Kaggle T4 x2 (sm_75, fp16, 30 GPU-h/week,
  12 h sessions). Never P100: sm_60, and Triton needs sm_70+. A local GPU is for
  short debug runs only, not training.
- **Training runs unattended** through the chain (`docs/kaggle.md`, "The chain"):
  add a run to `configs/chain/runs.yaml` on `master` and push. Runs pin a
  commit, so model code must be on `master` before a real arm is queued.
- **Real arms also need** the tokenized dataset uploaded to Kaggle once
  (`docs/kaggle.md`).
