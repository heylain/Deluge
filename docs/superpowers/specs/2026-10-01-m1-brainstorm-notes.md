# M1: brainstorm notes

> **Superseded by `2026-10-01-m1-design.md`**, the spec. This file is kept as the
> record of how each decision was reached; where the two differ, the spec wins.

Date: 2026-10-01 · Status: brainstorming complete; the spec is written.

**Handoff.** The original plan was to write the spec in a cloud session. In the
end, brainstorming finished locally on 2026-10-01 and the spec was written there
(`2026-10-01-m1-design.md`). The implementation plan follows from the spec.

## Decided

**1. M1 is the spec's M1, end to end** (`cfc-hybrid-spec.md` §13): build the
model, get A0 training stably against a matched tiny transformer, run the A1
arms through the Kaggle chain, and make the go/no-go call. If CfC is not better
than the gated conv within noise, text work freezes and the architecture
retargets to audio/event tokens (§11).

**2. Order: A0 before the Triton scan kernel.**

1. Model with a **head-batched chunked scan in plain PyTorch**.
2. A local smoke run per arm. This replaced the Kaggle throughput probe; see §3
   under Design.
3. A0 stability run.
4. A1 arms. There is no Triton scan in M1; see §3 under Design.

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

**The guard** (superseded by §3 below: no Triton in M1, since time is not a cost). `configs/train/screen.yaml`'s "~6 h per arm" assumes ~23k tok/s
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

**4. The noise rule, and a binding verdict at the dev budget** (decided
2026-10-01, revised the same day). It is fixed now, before any result exists,
so it cannot be fitted to the outcome. The metric is held-out loss in nats at
the end of a run's budget.

*Why it was revised.* The first version was adaptive and verdicted A1 at the
500M-token screening budget, to save weeks. The user then said time is not a
cost: runs are autonomous, and the project is deliberately slow. With weeks
free, the larger weakness is the budget, not the seed count. 500M tokens is 0.22
tokens/param, and rankings that early can flip. A k3 conv learns local
statistics fast, while using long-range state is learned later, so a short
screen errs toward a false no-go, which is the outcome that pivots the project.

**Week 1: screening, 500M tokens, ~30 h** at the unmeasured ~6 h/arm, queued in
this order:

1. CfC, seed 1234. This run is A0.
2. The all-attention transformer, seed 1234.
3. CfC, a second seed.
4. Gated conv (B3), seed 1234.
5. LRU (B4), seed 1234.

`seed` sets both init and data order (`TokenStream` takes `cfg.seed`), so arms
on the same seed see identical batches. That makes the comparison paired at no
cost. A second seed changes both, which overstates the noise slightly: the bar
gets stricter, not looser.

- **noise** = max(|CfC seed 1 − CfC seed 2|, 0.01 nats). This estimates the
  standard error of a difference between two single runs. The floor stops a
  lucky close pair from setting the bar too low.
- **A0 passes** if training is stable and CfC is within 1× noise of the
  transformer, or better. Otherwise stop and debug before going further.
- **Early A1 read**: conv − CfC at seed 1234, reported against the same noise.
  It is not binding and triggers no extra runs.
- LRU is not part of the go/no-go. With no MoD at A1, Δ is always 1, so CfC
  and LRU differ by the short conv only: the LRU arm measures the conv. Δ's
  worth is A3's question.

**Weeks 2–5: the binding A1 verdict, dev budget** (`configs/train/dev.yaml`,
2.26B tokens, ~20 tokens/param, ~27 h/arm at the unmeasured rate):

1. CfC, seed 1234.
2. CfC, the second seed.
3. Gated conv, seed 1234.
4. Gated conv, the second seed.

Each conv run is paired with the CfC run on the same seed.

- noise = max(|CfC seed 1 − CfC seed 2|, 0.01 nats), measured at this budget.
- gap = mean(conv) − mean(CfC); positive means CfC is better.
- **Go** iff gap ≥ 1.4× noise. That is two standard errors for 2-vs-2 means:
  the SE of the mean gap is σ = noise/√2, and 2σ = √2·noise ≈ 1.41× noise.
  Otherwise **no-go**: pivot per §11. No extension runs.

Prerequisite: a tokenized dataset of at least 2.26B training tokens plus a
held-out slice. The current upload is 600M; `docs/kaggle.md` already notes that
a dev arm needs ~4.5 GB.

**5. Every M1 run trains at seq_len 4096, on long documents** (decided
2026-10-01). As configured, A1 could not test §4's claim:
- Attention sits at blocks 4, 8 and 12 with W = 512. Stacked, that reaches about
  1,500 tokens back, in both arms.
- At `seq_len` 1024, the conv arm therefore already has a path to the whole
  sequence, and A1 would test only whether a recurrence is a better *local*
  mixer, which LFM2 found it is not.

The change:
- `seq_len` 4096 (§7's pretraining length) for every M1 run, including the
  transformer baseline, whose window becomes 4096.
- Data: only documents of at least 4,096 tokens, written as whole 4,096-token
  pieces with the remainder dropped, so each training sequence is a single
  document. Positions 1.5k–4k are then reachable by CfC state only.
- Metrics: the binding metric stays overall held-out loss under the decision-4
  rule. Held-out loss by position bucket ([0, 512), [512, 1.5k), [1.5k, 4k)) is
  reported as a diagnostic of where any gap comes from.
- Cost: about +17% FLOPs for the hybrid and +70% for the all-attention baseline
  at 4096 with an explicit-mask SDPA. That is calendar time only.
- Open for the plan: whether fineweb-edu sample-10BT holds 2.26B+ tokens in
  documents of at least 4k tokens. Fall back to sample-100BT.

## Design, agreed so far

**§1 Scope** (agreed 2026-10-01).
- **M1 builds:**
  - the model on PyTorch kernels;
  - the head-batched scan;
  - a held-out eval and the §7 stability monitors;
  - the configs: A1 arms re-based on A0, the transformer baseline, and the
    chain entries;
  - the probe, A0, A1, and a go/no-go writeup in `docs/results/`.
- **Deferred:**
  - chunk-summary attention, the summary-mask oracle, MoD, MTP, and the
    global-attention diagnostic go to M2;
  - the decode engine, the KV cache, and model-level train/infer equivalence go
    to M4;
  - the Triton scan happens in M1 only if the probe demands it, and is
    otherwise an M3 precondition.
- **Kept in M1:** a per-mixer test that running a sequence split in two,
  carrying state across the split, gives the same result as running it whole.
- **Milestone numbering:** §13 is the one numbering. M0 becomes what is done
  (repo, config, harness, chain, reference scans). The chain spec's "M2's
  `deluge.model:build`" becomes M1. Git history stays as it is.

**§2 Model code** (agreed 2026-10-01, amended after a review).
- **Mixer interface:** `forward(x, state=None, delta=None) -> (y, state)`.
  `state=None` means a fresh sequence from the learned init; `delta=None` means
  Δ=1. Attention accepts only `state=None` (its KV cache is M4).
- **The four mixers:**
  - CfC is exactly §3.
  - Gated conv is LFM2-style: `B, C, x̃ = W_in x`, then
    `y = W_out(C ⊙ DWConv_k3(B ⊙ x̃))`.
  - LRU is CfC without the conv and Δ.
  - Attention is the hybrid's own layer: GQA, RoPE θ=1M, 4 position-free sinks
    prepended to K/V, and an explicit SDPA mask. The mask is needed because
    `is_causal` would be off by n_sink with sinks prepended.
- **`mixer: {type: attention}`** has no fields of its own. In `config.py` it has
  `state_width = 0` and `uses_delta = False`, and `ModelConfig.params()` costs
  its slots with `_attention_layer_params()`.
- **Block and Model:** pre-norm residual `Block`. `Model` is embed → blocks →
  final RMSNorm → tied unembedding → CE in fp32, returning `(loss, metrics)`.
- **Transformer baseline:** A0 + `mixer: {type: attention}` + `window: 1024`
  (full causal) + `d_ff: 2408`. That is +0.05% against A0, all 12 layers
  identical, with GQA 12/4.
- **Gate bias `b_g`**, added to both `CfCMixer.params` and `LRUMixer.params`
  (the LRU shares the gated readout). It is initialised to 1.278, where
  SiLU = 0.9995 with gradient 1.000, and is decay-free. The spec's "gate starts
  near 1" cannot be met without it: on zero-mean input the mean gate is 0.07.
- **Init:**
  - `W_g` and `W_f` start at zero, so at step 0 the gate is exactly 1 and the
    decays are exactly the designed grid.
  - Both convs start as the identity on the current-token tap (the spec's
    "centre tap", read causally), so A1 does not also compare init schemes.
  - h₀ and the conv buffers start at zero.
  - Linears use std 0.02, and output projections are scaled by 1/√(2·12).
- **Decay init: λ log-uniform** (decided 2026-10-01). It is a deterministic
  geometric grid over heads from −ln a_max to −ln a_min, which is Mamba's dt
  convention. Memory lengths 1/λ: 999, 655, 429, 281, 184, 120, 79, 52, 34, 22,
  14, 9. The spec's literal wording (log a uniform) would space λ linearly and
  give one long head (999, 95, 50, …). Fix the wording in spec §3 and in the
  `_check_decay_range` docstring. The same grid applies to the LRU.
- **Weight decay:** the model's `no_weight_decay()` returns parameter names. The
  trainer falls back through `getattr` so stub factories work. In practice it
  only moves the conv buffers and sinks, since everything else on the list is
  1-D already.
- **Param accounting:** add the final RMSNorm to `ModelConfig.params()`, and test
  that the instantiated model's parameter count equals `cfg.params().total` for
  all four arms.

**§3 Scan, probe, guard** (agreed 2026-10-01, revised for "no rush").
- **`deluge/kernels/head_scan.py`:**
  - `head_chunked_scan(log_a [B,T,H], bv [B,T,D], h0 [B,D], chunk)`.
  - SSD-style. The intra-chunk work for all chunks is one batched matmul
    against a decay matrix `[B,H,N,C,C]` = exp(L_t − L_s), masked causal. The
    inter-chunk work is a 16-step loop over chunk-end states `[B,H,P]`. Then
    `states = within + exp(L_t)·h_chunk_start`.
  - fp32 inside, with autocast off, and differences of logs only.
  - About 50 MB per layer at 4 sequences per GPU.
  - `log_a = −Δ·λ` per token, so MoD gaps need no change later.
  - Tests: parity with `sequential_scan` forward and backward (per-head log_a
    expanded with `repeat_interleave`), a ragged final chunk, chunk sizes
    {1, 16, 64}, and the long-chunk underflow case.
  - No kernel registry until a second implementation exists. The mixers import
    the scan directly, which is fine because it is pure torch.
- **No Triton in M1.** The guard existed only to save weeks, and time is not a
  cost. Triton is an M3 precondition, where rented H100 time makes speed cost
  money.
- **No separate Kaggle probe.**
  - A local smoke run per arm on the 5070 Ti (16 GB, the same as a T4) settles
    `micro_batch` before anything is queued.
  - The harness logs peak GPU memory next to `tokens_per_second`.
  - A0's first session gives the real T4 numbers, and an OOM halts the chain
    within minutes.

**§4 Data, eval, monitors, "stable"** (drafted 2026-10-01; revised after a
review that found the off-by-one below).

- **Data: pieces of 4,097 tokens.**
  - Why: `TokenStream.batch` reads `seq_len + 1` tokens (`stream.py:122`), so
    4,096-token pieces would give each document's last piece a target from the
    next document, inside the [1.5k, 4k) bucket. Also, `(len − 1) // seq_len`
    drops a file's last whole piece.
  - `TokenStream` gets a piece mode with a step of `seq_len + 1`.
  - The data is a directory of `train.bin`, `heldout.bin` and one
    `provenance.json`. `heldout.bin` and the provenance file are found next to
    `--data`, so the chain schema does not change. `prepare_data.py` names its
    outputs explicitly instead of using `with_suffix(".json")`.
  - Guard: `piece_len == seq_len + 1`. With no provenance file (synthetic data,
    the rehearsal), the stream runs in today's contiguous mode.
- **Split.**
  - A blake2b hash of each document's text sends ~1% of documents to held-out:
    about 24M tokens, and a forward pass over it costs ~0.35% of a dev run.
    Exact duplicates land on the same side, which matters because FineWeb is not
    deduplicated across dumps.
  - `heldout.bin` is written in hash order, so its first 2M tokens are a random
    subset of documents.
  - The provenance file records the dataset revision.
- **Yield first.** The plan's first data task builds a histogram of
  fineweb-edu's `token_count` column (GPT-2 tokens) over one shard, before any
  size is fixed. The same column pre-filters short documents before tokenizing.
  If the yield is short, use sample-100BT.
- **Eval.**
  - It runs through `forward(x, y, reduction="none")`, so DataParallel splits
    it. That returns per-token losses `[B,T]` and the monitor stats shaped
    `[B, …]` (mean λ per head per layer, h RMS).
  - Off when there is no `heldout.bin` or `eval_every_tokens` is null.
  - The cadence is 50M tokens for screening and 100M for dev, on the 2M subset.
    Evals land on checkpoint steps (190 = 2 × 95; 381), and the eval runs before
    the save.
  - `log.jsonl` readers deduplicate eval records by step, last one wins.
  - `eval_every_tokens` stays out of the fingerprint.
- **Final pass.** It computes the full-set loss (the binding A1 metric) and the
  subset loss (for stability criterion 2). `final_eval.json` is written
  atomically (temp file + `os.replace`) and is the completion marker: exit 0 only
  once it exists. A resume after the budget is spent runs only the eval.
- **Tests.**
  - `test_resume` proves eval calls do not move the step count or the data
    position.
  - A new torch-level test in `test_torch_backend.py`: N steps with an eval at
    every step and N steps with none give bit-identical weights and optimizer
    state.
- **"Stable"**, the first half of A0's pass test:
  1. Final held-out loss is finite.
  2. Final subset loss ≤ the best in-run subset loss + noise.
  3. No layer has all heads at a < 0.5 or all at a > 0.9999. Here a =
     exp(−mean λ), the geometric mean over held-out tokens. Above 0.9999 is a
     memory of about 10k tokens, longer than the sequence. λ is unbounded both
     ways, so both thresholds can fire.
  4. At most 1% of steps are skipped by the fp16 scaler. A skip is detected as
     the scale dropping across `scaler.update()`. The skip count is reported
     with the A1 results whatever it is.
- **micro_batch** is split across the two T4s by DataParallel. The local smoke
  run gives a per-GPU value, and Kaggle uses double it, which must be even. The
  stale comments in `base.yaml` ("256 sequences of 1024", "32 accumulation
  steps") get updated.

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
