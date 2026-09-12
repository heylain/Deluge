# CfC–Attention Hybrid — Architecture & Engineering Spec

**Status:** draft v0.2 — 2026-09-11
**Author:** seyncia
**Codename:** Deluge (working; check GitHub/HF for collisions before the repo exists)

---

## 0. Thesis

A decoder-only language model whose bulk is parallel-scannable Closed-form Continuous-time (CfC) recurrent layers, with a small number of attention layers for exact retrieval. Every design choice is judged by one number: **bytes read per generated token at batch 1 on a 16 GB consumer GPU.** Training runs on remote compute; inference is the constraint.

Two claims the project exists to test:

1. A gated CfC recurrence beats a gated short convolution as the "cheap" layer type once a few attention layers handle long range. (Liquid AI's hardware-in-the-loop search found the opposite for SSMs; this project's answer is Δ-aware decay + Mixture-of-Depths gaps, see §4.)
2. Chunk-summary attention over recurrent state gives retrieval quality close to full attention with an O(L/C) cache.

Non-goals: matching frontier quality, multimodal input, training-time throughput records, a thinking mode in v1 (see §14 — v1 is the fast model; effort tiers and reasoning are a follow-on project on the same weights).

---

## 1. Constraints

| | |
|---|---|
| Training | remote cluster, bf16, FSDP/DDP; budget set in §7 |
| Inference | single 16 GB GPU (RTX 5070 Ti), batch 1, decode-latency first |
| Memory model | decode is memory-bandwidth-bound → minimise active weight bytes + state bytes; ~896 GB/s ceiling |
| Contexts | 32k native target; state-carried "infinite" streaming for recurrent layers |
| Runtime | custom PyTorch + Triton; no dependency on llama.cpp/vLLM (they cannot express the scan, MoD skip, or state rollback) |

---

## 2. Architecture overview

Pre-norm residual stack, RMSNorm everywhere. Repeating unit:

```
[CfC-mix → FFN] [CfC-mix → FFN] [CfC-mix → FFN] [Attn → FFN]
```

- 3:1 recurrent:attention ratio (Griffin / Jamba / Nemotron-H / LFM2 range).
- Attention layers are **sliding-window + chunk-summary** (§5). One attention layer per 4 units is made global for the dev config only, as a diagnostic.
- Mixture-of-Depths (§6) is applied to every second unit's CfC-mix block and its FFN.
- FFN is dense SwiGLU in v1; MoE in v2 (§8), swapped in without changing anything else.
- Tied input/output embeddings.

### 2.1 Reference configs

| | Dev (single GPU) | Target v1 (dense) | Target v2 (MoE) |
|---|---|---|---|
| d_model | 768 | 2048 | 2048 |
| units × pattern | 3 × (C,C,C,A) = 12 blocks | 6 × (C,C,C,A) = 24 blocks | same |
| CfC d_inner | 1152 (1.5×) | 3072 | 3072 |
| CfC heads (decay groups) | 12 | 16 | 16 |
| Attn heads / KV heads | 12 / 4 (hd 64) | 16 / 4 (hd 128) | same |
| Window W | 512 | 2048 | 2048 |
| Chunk C | 64 | 64 | 64 |
| FFN | SwiGLU d_ff 2048 | SwiGLU d_ff 5632 | 16 experts d_ff 1408, top-2 + shared d_ff 2816 |
| Vocab | 32k | 32k | 32k |
| MoD capacity ρ | 0.5 | 0.5 | 0.5 |
| Params total / active | ~113M | ~1.3B | ~4.2B / ~1.3B |
| bf16 weights | 0.23 GB | 2.6 GB | 8.4 GB |
| int4 weights | — | 0.7 GB | ~2.4 GB |

Parameter counts recomputed from this table (tied embeddings; learned h₀, conv buffers and sink KV included): dev 113M, v1 1.331B, v2 4.239B total / 1.332B active — v2 is sized so its *active* params equal v1's to within 1M, making the MoE ablation clean. The dev row previously read ~160M; that was an overestimate, and the corrected number is what the matched tiny-transformer baseline in M0/A0 should be sized to (it also moves dev's ~20 tokens/param floor to ~2.3B tokens). Re-derive all three from the instantiated model once `config.py` exists — before any remote-compute commitment (§7).

---

## 3. CfC-mix layer

Input x_t ∈ ℝ^d. All gates depend on the **input only** — this is what makes the recurrence linear in h and therefore parallel-scannable. State-dependent time constants (the original "liquid" property) are deliberately dropped; the continuous-time property survives through Δ_t. Recorded as [ADR-0001](decisions/0001-input-only-gates.md); the scan kernel, MoD Δ-gaps and speculative rollback all assume it.

```
v_t   = SiLU( DWConv_k4( W_v x_t ) )                 candidate, ℝ^{d_inner}
λ_t   = softplus( W_f x_t + b_f )                    decay rate, ℝ^{H}, broadcast over head channels
a_t   = exp( −Δ_t · λ_t )                            decay ∈ (0,1)
b_t   = sqrt( 1 − a_t² )                             input scale (Griffin normalisation, bounded state)
h_t   = a_t ⊙ h_{t−1} + b_t ⊙ v_t                    scan
y_t   = W_out ( RMSNorm(h_t) ⊙ SiLU( W_g x_t ) )     gated readout
```

Correspondence to CfC: with h-independent heads, CfC's σ(−f·t) interpolation reduces to the exponential-decay form above; f ↔ λ, the "time" argument ↔ Δ_t.

**Δ_t semantics.** Δ_t = 1 for consecutive processed tokens. When MoD skips this block for one or more tokens, Δ_t for the next processed token = 1 + number of skipped tokens (§6). For irregularly sampled inputs (audio/event streams, future work) Δ_t = real time gap.

**Init.** λ initialised so that a_t at Δ=1 is log-uniform in [0.9, 0.999] across heads (λ ∈ [0.001, 0.105]); set b_f accordingly. W_g init so the gate starts near 1. DWConv init identity on the centre tap. This init is the single most important stability lever.

**Initial state h₀.** A learned parameter per layer (ℝ^{d_inner}, zero-init, weight-decay-free), not a zero constant. See §15 — this is the recurrent half of the baked-in system prompt. The conv buffer's initial contents are likewise learned (3 × d_inner per layer).

**Numerics.** Store and compute λ (log-domain), never a_t directly; a_t is materialised inside the kernel in fp32. Precision near a=1 matters: 0.999 vs 0.998 halves the time constant.

**Kernels.** Training: chunked parallel scan (GLA/Mamba2-style, chunk 64–128) in Triton; within-chunk parallel, cross-chunk sequential state passing. Inference: fused single-step kernel (decay + update + norm + gate + W_out in one launch).

**Expressivity note.** Diagonal positive decays cannot represent some state-tracking (parity etc.). Accepted; attention layers cover retrieval, and negative-eigenvalue variants are an ablation, not a v1 feature.

---

## 4. Why CfC over gated short-conv (the claim to defend)

Liquid AI's search found kernel-3 gated convolutions match or beat SSMs when GQA layers are present. This project's bet is that the recurrent layers earn their keep in two regimes the conv cannot reach:

1. **Long horizon without attention**: state carries information across arbitrarily long spans; conv sees 3 tokens.
2. **Δ-aware decay under MoD**: skipped tokens become real time gaps; the recurrence integrates over them coherently, a conv cannot.

If ablation A1 (§11) shows no gain over gated conv on text, the architecture pivots to audio/event tokens where Δ_t is native.

---

## 5. Attention layer: sliding window + chunk summaries

Standard GQA with RoPE (θ = 1M), sliding window W over recent tokens, **plus** one summary KV pair per completed chunk of C tokens for the whole context.

**Summary source.** At the end of chunk j (token position p_j = (j+1)·C − 1), take the residual-stream hidden state x_{p_j} entering this attention layer, concatenated with the RMSNormed CfC state h_{p_j} from the immediately preceding CfC-mix layer; project with W_sk, W_sv to a K/V pair per KV head. Position for RoPE = p_j (the chunk's last token), so summaries and window tokens live in one positional frame.

**Mask (exact, identical in train and inference).** Token t attends to:
- window tokens s with t − W < s ≤ t, and
- summary j for every chunk with p_j < t − W (chunks fully outside the window). Chunks overlapping the window are seen through their raw tokens only — no double counting.

**Cache at inference.** Ring buffer of W KV pairs + growing list of ⌊L/C⌋ summary pairs → O(W + L/C). At 32k, C=64: 512 summaries.

**Learned sink KV pairs.** Each attention layer has n_sink = 4 learned K/V pairs per KV head that are always present in the cache and never evicted from the window. They serve as attention sinks (stability) and as the attention half of the baked-in system prompt (§15). They carry no RoPE (position-free).

**Global diagnostic.** Dev config only: one attention layer with full causal attention, used to measure how much retrieval the summary path loses.

---

## 6. Mixture-of-Depths (MoD)

Applied to every second (C,C,C,A) unit's CfC-mix + FFN blocks (attention blocks never skip in v1).

**Training routing.** Scalar router r_t = w_r · RMSNorm(x_t). Top-⌈ρ·L⌉ tokens per sequence are processed; the block output is scaled by σ(r_t) so the router receives gradient. Others pass through the residual untouched. This top-k over the sequence is non-causal and is **training-only**.

**Causal predictor for inference.** A tiny head (linear → sigmoid) trained with BCE to predict "in top-k" from x_t alone; at inference, process iff predicted p > 0.5 (threshold tunable per block for speed/quality). Predictor loss weight 0.01, stop-gradient into the trunk.

**Interaction with CfC.** A skipped token does not advance h. The block keeps a per-sequence counter g (tokens since last processed); the next processed token uses Δ_t = 1 + g. This is the mechanism that makes continuous time meaningful on text.

**Interaction with FFN/MoE.** A token skipping the CfC-mix block also skips the unit's FFN. In v2 this compounds sparsity: ρ × (active experts / total).

---

## 7. Training

**Data.** Web + code + a small high-quality replay set; exact mixture TBD. Deliberately over-weight reasoning-shaped text (math, code, worked step-by-step solutions, argumentative prose) even though v1 has no thinking mode — models post-train into thinking far more easily when pretraining saw the shape of it (§14).

**Tokenizer.** 32k BPE (reuse an existing 32k tokenizer to save a subproject; the small vocab keeps the unembedding GEMV cheap at decode). Reserve special tokens now — all of them unused in v1, none of them ever emitted (§14):

| Token(s) | Reserved for |
|---|---|
| `<think>`, `</think>` | thinking mode (§14) |
| `<effort:0>` … `<effort:3>` | effort tiers (§14) |
| `<tool_call>`, `</tool_call>` | tool invocation (follow-on) |
| `<tool_result>`, `</tool_result>` | tool output returned to the model |
| 8 unnamed spares | whatever the follow-on needs |

Eighteen ids. Claim them from the reused tokenizer's existing unused/reserved slots where it has them rather than extending the vocab — extending changes the embedding and unembedding shapes for nothing. Allocate the reserved block contiguously so it is trivial to mask at sampling time in v1. Retokenizing later is far more painful than reserving eighteen ids today.

**Token budget (target v1, 1.3B dense).** Minimum ~26B tokens (~20 tokens/param); preferred 50–100B for inference-favourable overtraining. Rough cost at 50B tokens: 6·N·D ≈ 3.9e20 FLOPs ≈ 270 H100-hours at 40% MFU → e.g. 8×H100 for ~35 h. v2 MoE: same active FLOPs, ~3× memory per rank; needs expert parallel or 8 × 80 GB.

**Losses.**

```
L = L_ce
  + 0.3 · L_mtp            (depth-1 MTP, annealed to 0.1 after 60% of training)
  + 0.01 · L_mod_pred      (causal skip predictor BCE)
  + 1e-3 · L_z(mod router) (router logit z-loss)
  + [v2] 1e-3 · L_z(moe router); load balance via per-expert bias, no aux loss
```

**MTP head.** DeepSeek-style: one extra CfC-mix + FFN block, shared embedding/unembedding, input = W_p [RMSNorm(x_t^{L}); RMSNorm(emb(x_{t+1}))]. Adds ~1 block of params. Depth 2 is an ablation.

**Optimizer / schedule.** AdamW β=(0.9, 0.95), wd 0.1 (no decay on decays λ, norms, biases). Peak LR 3e-4 (dev: 6e-4), 2k warmup, cosine to 10%. Grad clip 1.0. bf16 autocast, fp32 master + scan internals.

**Sequence length.** 4k for the first 80%, extend to 32k for the last 20% with RoPE base scaling on attention layers; recurrent layers need no change. Chunk-summary mask makes 32k sequences memory-cheap even in training.

**Stability watch-list.** Decay collapse toward a→1 (state blow-up) or a→0 (amnesia); MoD router saturation (all-skip / no-skip); summary-attention leakage (test the mask against a brute-force reference on every commit).

---

## 8. FFN: dense → MoE (v2)

Fine-grained MoE, DeepSeek-V3 style:

- 16 experts, each SwiGLU d_ff 1408; top-2 routed + 1 always-on shared expert d_ff 2816.
- Sigmoid router, renormalised top-k weights, per-expert bias updated each step to balance load (no aux loss), z-loss 1e-3.
- Dropless: sort tokens by expert, grouped GEMM (torch grouped mm / Triton).
- Inference: only active experts' weights are read per token — the direct bandwidth win. Total 4.2B fits at int4 (~2.4 GB) with room for a 32k context cache.

Swap-in rule: v2 must keep every other hyperparameter identical to v1 so A5 (§11) is a one-variable ablation.

---

## 9. Inference runtime

A standalone decode engine; this is a project in itself.

**Weights.** int4 AWQ-style group quantisation (group 128) with fused dequant-GEMV kernels; attention and CfC projections may stay int8 if quality drops. Embedding/unembedding int8.

**Per-sequence state struct.**

```
for each CfC layer:   h (d_inner, fp32), conv_buf (3 × d_inner), mod_gap counter
for each Attn layer:  kv_ring (W × 2 × n_kv × hd, int8 keys per-channel / values per-token), summaries (list)
misc:                 position, chunk_progress, rng
```

Whole struct at target v1 ≈ a few hundred KB excluding the KV ring; copy = snapshot. A fresh sequence is initialised from the learned h₀ / conv buffers / sink KV (§15), not from zeros.

**Snapshot API.** `snapshot() → bytes`, `restore(bytes)`. Uses: prefix caching (system prompt processed once, zero prefill per session), shipping personas as state files, conversation rewind, and rollback for speculative decoding.

**Speculative decoding.** MTP head drafts k tokens (k=2–4). Verify with one batched forward of k+1 tokens. Because the recurrence cannot be inverted, take a snapshot at each draft position during verification and restore the last accepted one; attention caches truncate as usual.

**Execution.** One CUDA graph per decode step (fixed shapes; MoD skip implemented as masked compute inside the graph rather than dynamic control flow, or as two captured graphs per block). Target: <0.5 ms launch overhead per token. Prefill uses the chunked scan and chunk-summary mask.

**Throughput expectation (v1, int4, batch 1).** Weight bytes/token ≈ 0.7 GB × (1 − ρ/2 effective skip) → bandwidth ceiling ~1500+ tok/s; realistic target 400–800 tok/s after kernel work. Eager PyTorch will show <100 tok/s; that gap is the runtime project's scoreboard.

---

## 10. Evaluation

**Baselines (same tokens, same tokenizer, same schedule):**
- B1: dense transformer, matched active params.
- B2 (v2 only): dense transformer, matched total params.
- B3: this architecture with CfC-mix replaced by LFM2-style double-gated kernel-3 conv.
- B4: this architecture with CfC-mix replaced by a plain LRU (no Δ, no MoD gap).

**Quality.** Held-out perplexity by domain; standard small-model suite (HellaSwag, ARC, PIQA, WinoGrande, GSM8K-lite); needle-in-a-haystack at 4k/16k/32k with the needle placed inside vs outside the window (isolates the summary path); multi-needle and passkey-with-distractors; and a variant where the needle sits in *model-generated* text (long continuation then recall), since the future thinking mode stresses the summary path during generation, not prefill.

**Speed.** Decode tok/s at batch 1 (eager, compiled, CUDA-graphed); prefill tok/s at 2k/16k/32k; latency breakdown per layer type; state/cache bytes vs context length.

**Reporting rule.** Any speed number without the matched-baseline number next to it does not go in the writeup.

---

## 11. Ablation ladder (each step changes one thing)

| ID | Change | Question answered |
|---|---|---|
| A0 | Dev config, dense FFN, no MoD, window-only attention | Does the scan train stably? |
| A1 | CfC-mix vs gated conv (B3) vs LRU (B4) | Is CfC worth having on text at all? |
| A2 | + chunk-summary attention | How much retrieval survives O(L/C) cache? |
| A3 | + MoD with Δ-gap coupling; ablate Δ-gap → Δ=1 | Does continuous time do work under MoD? |
| A4 | + MTP head | Draft acceptance rate vs quality cost |
| A5 | dense FFN → MoE (v2) | Quality at matched active; decode speedup at int4 |
| A6 | negative-eigenvalue decays; depth-2 MTP; ρ sweep | Nice-to-haves |

Go/no-go after A1: if CfC ≤ conv within noise, freeze text work and re-target the architecture to audio/event tokens.

---

## 12. Risks

- **A1 fails** — mitigated by the pivot rule above; the runtime, MoD and summary-attention work carry over regardless of layer type.
- **MoD non-causality bugs** — train-time top-k must never leak into inference; the causal predictor is the only inference path. Unit-test that inference outputs match teacher-forced training routing on held-out sequences within tolerance.
- **Summary-attention mask leakage** — brute-force reference test in CI.
- **Rollback correctness** — golden test: greedy decode with speculation must equal greedy decode without it, token for token.
- **Long-context decay range** — a=0.999 ≈ 1k-token memory; MoD gaps extend effective horizon, but genuine 32k dependencies are the attention path's job. Do not oversell recurrent long-range recall.
- **Runtime scope creep** — the decode engine is the real cost. Time-box: kernels for scan + fused CfC step + int4 GEMV first; everything else can be plain PyTorch initially.

---

## 13. Milestones

1. **M0** — Repo, config system, chunked scan kernel + brute-force reference tests. A0 trains on the dev config to the same loss as a matched tiny transformer.
2. **M1** — A1 result. Go/no-go.
3. **M2** — Chunk-summary attention + MoD + MTP implemented; A2–A4 at dev scale. Also sweep the §16 questions tagged **M3** (n_sink, MTP block sharing, effort-token conditioning of the MoD predictor) — they freeze at pretraining time, so this is the last milestone that can answer them.
4. **M3** — Target v1 remote run (50B tokens). Baselines B1/B3 trained alongside. Preconditions: param counts and compute budget re-derived from `config.py` (§2.1), and every §16 **M3** question answered.
5. **M4** — Decode engine: int4, CUDA graphs, snapshots, speculative rollback. Speed numbers vs B1.
6. **M5** — v2 MoE run and A5.
7. **M6** — Writeup + release.

---

## 14. Thinking mode and effort tiers (follow-on project, planned for now)

v1 ships without a thinking mode. A pretrained model has none by definition; it's a post-training behaviour. The follow-on ("Deluge-think") adds it on the same weights. What v1 does now so that project needs no retraining:

- Reserved `<think>`/`</think>`/`<effort:n>` tokens (§7).
- Reasoning-shaped pretraining mix (§7).
- Effort tiers are defined as **inference knobs the architecture already exposes**, before any RL exists:

| Tier | MoD predictor threshold | MoE top-k (v2) | Spec draft length | Thinking budget |
|---|---|---|---|---|
| 0 — instant | high (aggressive skip) | 1 + shared | 4 | none |
| 1 — fast | default 0.5 | 2 + shared | 3 | none |
| 2 — think | low (process most) | 2 + shared | 2 | bounded |
| 3 — deep | ~0 (process all) | 4 + shared | 1 | large |

Tiers 0–1 are usable in v1 as pure speed/quality dials. Tiers 2–3 acquire meaning after post-training. Effort tokens become the conditioning signal for the policy in the follow-on; in v1 they are never emitted.

Post-training plan (sketch, not v1 scope): SFT on traces → RL with a length-aware reward, effort token as a control input. Feasibility at this size is established (Liquid AI obtained a thinking mode from a 2.6B hybrid via RL).

Architectural consequence for v1: thinking traces are long *generated* contexts. The chunk-summary attention path must be evaluated on generated text (§10), not only on prefilled prompts.

## 15. Baked-in system prompt

Three mechanisms, cheapest to most permanent. v1 implements the first two; the third belongs to the follow-on.

1. **Learned initial state (pretraining, ~free).** Each CfC layer's h₀ and conv buffer are learned parameters (§3); each attention layer has n_sink learned KV pairs permanently in the cache (§5). Together this is a system prompt in recurrent form — a few hundred KB the model "reads" before token one, at zero inference cost. Trained jointly from step 0. Later: fine-tune only these tensors (prefix-tuning on state) to produce swappable personas shipped as state files.
2. **Snapshot of a text prompt (no training).** Run any system prompt once, `snapshot()`, ship the bytes. Editable in plain text; only as strong as the model's instruction-following.
3. **Post-training internalisation (permanent, not swappable).** SFT/RL so the default behaviour holds with no prompt at all. Lives with the effort-tier work.

**Caveat.** A learned h₀ decays at the same rate as everything else in the recurrent layers (~1k tokens at a=0.999; MoD gaps extend this somewhat). Behaviour that must persist across a long conversation has to come from the sink KV pairs (never evicted) or from mechanism 3 — not from h₀ alone. Measure this: run the persona-consistency probe at 512 / 4k / 32k tokens into a conversation.

**Runtime rule.** A fresh sequence = learned init (1) + optional restored snapshot (2). Snapshots are taken *after* the learned init, so they compose.

## 16. Open questions

Each question carries a **decide-by** tag:

- **M0** — the answer is an interface or tensor shape the first code bakes in; decide before writing model code.
- **M3** — frozen into the pretrained checkpoint and impossible to retrofit; sweep at dev scale during M1–M2 and settle before the target v1 run starts.
- **contingent** — blocked on a result that does not exist yet; answering early is work thrown away.

Nothing here is decidable from first principles the way ADR-0001 was. Anything not tagged **M0** stays a config knob, not a guess written down as a decision.

- **[M0]** Per-head vs per-channel decay λ (per-head is cheaper and matches Griffin; per-channel is closer to CfC). Decide the *shape*, not the value: λ ∈ ℝ^H already broadcasts over d_inner/H channels, so per-channel is the H = d_inner case of the same kernel. Make H a config value (invariant d_inner % H == 0) and sweep in A6. Note per-channel makes W_f d×d_inner — +113M on target v1, not free.
- **[M0]** Should summaries be written from every CfC layer's state or only the one preceding the attention layer? Fix the *interface* now: `model.py` collects a list of designated summary-source states and passes it to the attention layer, which sizes W_sk/W_sv from the list length. v1 populates it with one entry; "every layer" is then a config change, not a rewrite of the block plumbing.
- **[contingent]** MoD on attention blocks in v2? Revisit with A5; nothing in v1 depends on it as long as the MoD wrapper stays block-level.
- **[M3]** Whether to share the MTP block with the last trunk unit to save params. Decide at A4 on measured draft acceptance vs quality cost, before the target run's parameter budget is fixed.
- **[contingent]** Audio tokenizer choice if the pivot happens. Blocked on A1 failing (§11 go/no-go) — discarded work if A1 passes.
- **[M3]** n_sink: 4 is a guess; sweep 1/4/16 and check whether more sinks trade against window capacity. The sink KV pairs are learned per-layer parameters, so this freezes at pretraining time.
- **[contingent]** Should the learned h₀ be per-layer only, or per-layer-per-head with a shared low-rank basis (cheaper persona fine-tunes)? h₀ is tiny and separately fine-tunable (§15 mechanism 1), so the parameterisation can change post-hoc; belongs with the persona work.
- **[M3]** Whether effort tokens should also condition the MoD predictor at pretraining time (would make tier 0/1 behaviour learned rather than thresholded). Same class as the reserved token ids (§7): near-free to enable during pretraining, impossible to add to trained weights afterwards.
