# ADR-0001: All CfC-mix gates depend on the input only

**Status:** Accepted
**Date:** 2026-09-12
**Deciders:** seyncia
**Spec reference:** `../cfc-hybrid-spec.md` §3 (CfC-mix layer), §6 (MoD Δ-gap), §9 (rollback)
**Superseded by:** —

---

## Context

The CfC-mix layer is the bulk of the model. Its recurrence is

```
λ_t = softplus( W_f x_t + b_f )        decay rate, function of x_t ONLY
a_t = exp( −Δ_t · λ_t )
b_t = sqrt( 1 − a_t² )
h_t = a_t ⊙ h_{t−1} + b_t ⊙ v_t        v_t is also a function of x_t only
```

The original "liquid" formulation (LTC/CfC) makes the time constant a function of the
*state*: τ = τ(h, x). That is the property the name refers to, and it is the first thing
that will look tempting to reintroduce once the model is training and someone (me) wants
it to feel more like a real liquid network.

This ADR records that the gates are **deliberately** input-only, why, and what breaks if
that changes. It exists because the decision is easy to erode one commit at a time
(a small h-dependent term in λ "just for expressivity") and every downstream component
silently assumes it.

## Decision

Every quantity that multiplies or feeds `h_{t−1}` in the transition — `λ_t`, `a_t`, `b_t`,
`v_t` — is computed from `x_t` (and its short causal conv context) alone. The recurrence
is therefore **linear in h** with time-varying, input-dependent, but state-independent
coefficients.

State-dependent time constants are not a v1 feature, not a v1 ablation, and not to be
added under a config flag that defaults off. If they are ever tried, that is a separate
architecture with its own ADR, kernel, and MoD semantics.

## Why (what depends on this)

1. **Parallel scan (training).** `h_t = a_t h_{t−1} + u_t` with `a_t, u_t` known before
   the scan is an associative recurrence → chunked parallel scan (GLA/Mamba2-style,
   §3 Kernels). If `a_t` depends on `h_{t−1}`, the recurrence is nonlinear, there is no
   prefix-scan form, and training is a sequential loop of length L per layer. At 4k–32k
   sequences that is the difference between a project and not a project.

2. **MoD Δ-gap coupling (§6).** A token skipped by MoD does not update h; the next
   processed token uses `Δ_t = 1 + g`. This is exact only because
   `exp(−(1+g)·λ) = exp(−λ)^{1+g}` — i.e. because λ does not depend on the states the
   model never computed for the skipped tokens. With state-dependent λ, "integrate over
   the gap" has no closed form; the whole "continuous time does work under MoD" claim
   (A3) rests on this line.

3. **CfC correspondence (§3).** The reduction of CfC's σ(−f·t) interpolation to the
   exponential-decay form holds *because* f is h-independent. Continuous time survives
   through Δ_t; the state-dependence does not, and that is the trade.

4. **Speculative decoding verification (§9).** Verifying k+1 draft tokens in one batched
   forward uses the same chunked scan as training. State-dependent gates would force the
   verify pass to be sequential too, and the snapshot/restore invariant (greedy-with-spec
   == greedy-without, token for token) becomes much harder to keep exact.

5. **Train/inference equivalence.** Chunked scan (training, prefill) and fused single-step
   kernel (decode) compute the same function. Every reference test in M0 is written
   against a sequential scalar loop; those tests are only meaningful while the loop and
   the chunked kernel are algebraically the same thing.

6. **Griffin normalisation and numerics.** `b_t = sqrt(1 − a_t²)` bounds ‖h‖ given a_t is
   a known coefficient; λ is stored in log-domain and a_t is materialised in-kernel in
   fp32. Both assume a_t is a pure function of the current input.

7. **Chunk summaries (§5).** Summaries read `RMSNorm(h_{p_j})` at chunk boundaries and are
   produced inside the chunked scan at training time. Same dependency as (1).

## Consequences

**Accepted losses.**
- No state-dependent time constants. The model is closer to a gated LRU / Griffin RG-LRU
  with a Δ argument than to a textbook CfC. Name it honestly in the writeup.
- Diagonal positive decays cannot represent some state-tracking (parity etc.). Already
  accepted in §3; attention layers cover retrieval. Negative-eigenvalue decays are an
  ablation (A6) and are still input-only — they do not violate this ADR.

**Where nonlinearity and state-dependence still live** (so the urge to add them to λ has
somewhere else to go):
- The candidate `v_t = SiLU(DWConv(W_v x_t))` and the readout gate `SiLU(W_g x_t)`.
- The FFN after every mix block.
- **Across layers.** `x_t` for layer ℓ+1 is the residual stream, which already contains
  layer ℓ's `y_t` and therefore its state. Later layers' time constants *do* depend on
  earlier layers' state — one layer down, not within the recurrence. This is the
  "liquid" behaviour the stack actually has, and it costs nothing.

**Engineering rules that follow.**
- The gate function signature is `gates(x) -> (λ, v, g)`. It never takes h. A test asserts
  `∂λ_t / ∂h_{t−1} == 0` exactly (autograd on a small config) and stays in CI.
- Any PR touching `λ_t`, `a_t`, `b_t`, or the scan kernel signature links this ADR in the
  description.
- If A1 fails (§11 go/no-go) the answer is the audio/event pivot, not state-dependent
  gates. The pivot keeps every kernel; the gates change throws all of them away.

## Alternatives considered

- **Full CfC / LTC with τ(h, x).** Rejected: sequential-only, incompatible with (1), (2),
  (4). This is the original liquid formulation; it is the thing this ADR says no to.
- **One-step-delayed state feedback** (λ_t depends on `sg(h_{t−1})` or on `h_{t−1}`
  through a detached path). Rejected: still breaks the parallel scan (a_t needs h_{t−1},
  which needs a_{t−1}, ...). Detaching gradients does not make the forward pass
  parallel.
- **Per-channel vs per-head λ.** Orthogonal — both are input-only. Open question in §16,
  not governed here.
- **Data-dependent but *chunk-level* state feedback** (λ for chunk j+1 conditioned on the
  summary state at the end of chunk j). Not rejected; it keeps within-chunk parallelism
  and cross-chunk sequential passing, which the kernel already does. If state-dependent
  time constants are ever revisited, this is the only shape that fits the existing
  kernel, and it gets its own ADR.

## Revisit triggers

Reopen this decision only if all of the following hold:
- A1 shows CfC ≤ gated conv within noise **and** the audio pivot is also rejected;
- a sequential or chunk-sequential kernel has been benchmarked and its training cost
  accepted in writing;
- the MoD Δ-gap semantics have been redefined for the new transition.

Otherwise: it was tempting, it was considered, the answer is still no.
