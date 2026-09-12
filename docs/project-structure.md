# Project Structure

Package name used below: `deluge` (rename freely — it's one `sed`).

## Principles the layout enforces

1. **Three consumers, one model definition.** Training, the inference engine, and evals all import the same `model/` code. The model package must not import FSDP, Triton, or the data loader — only `torch`. Anything heavy is injected (kernels via a registry, parallelism via wrappers).
2. **Every kernel has a slow twin.** `kernels/reference/` is pure PyTorch, obviously correct, and is the source of truth. Triton kernels are only allowed to exist next to a parity test against their reference. The chunk-summary mask and the scan both live here, because those are where silent bugs hide.
3. **State is a first-class type.** The per-sequence state struct (`model/state.py`) is defined once and used by the training scan (chunk boundaries), the inference loop, snapshots, and speculative rollback. If train and infer disagree about what "state" is, the golden tests fail.
4. **Configs are data, experiments are scripts.** No hyperparameter lives in Python. Every ablation in the spec is a YAML diff against a base config, so the ladder is reproducible by name.
5. **The inference engine is a separate package with its own tests and benchmarks.** It's the expensive part and must be developable without the training stack installed.

## Tree

```
deluge/
├── README.md
├── pyproject.toml              # extras: [train], [infer], [dev]; base install is pyyaml + numpy
├── docs/
│   ├── cfc-hybrid-spec.md      # the architecture spec
│   ├── project-structure.md    # this doc
│   ├── kaggle.md               # free-tier runbook: T4x2, 12h sessions, resume
│   ├── decisions/              # ADR-style: one file per irreversible choice, dated
│   │   └── 0001-input-only-gates.md
│   └── results/                # ablation tables, committed as markdown + csv
│
├── configs/
│   ├── base.yaml               # every knob, documented inline
│   ├── model/
│   │   ├── screen.yaml         # 43M half-width, for harness work only
│   │   ├── dev.yaml            # 113M single-GPU
│   │   ├── target_v1.yaml      # 1.3B dense
│   │   └── target_v2.yaml      # 4.2B/1.3B MoE (inherits v1, overrides ffn:)
│   ├── train/                  # budgets; the model config is chosen separately
│   │   ├── base.yaml           # every knob, documented inline
│   │   ├── screen.yaml         # 500M tokens: ranks arms, does not train one
│   │   ├── dev.yaml            # 2.26B tokens, the ~20 tok/param floor
│   │   └── remote_v1.yaml
│   └── ablations/
│       ├── A0_scan_only.yaml
│       ├── A1_conv_baseline.yaml
│       ├── A1_lru_baseline.yaml
│       ├── A2_chunk_summary.yaml
│       ├── A3_mod_delta_gap.yaml
│       ├── A3_mod_delta_one.yaml
│       ├── A4_mtp.yaml
│       ├── A4_mtp_shared.yaml   # the spec 16 M3 sharing question, as two arms
│       └── A5_moe.yaml          # not yet: needs MoE dims derived at dev width
│
├── deluge/                     # the importable package
│   ├── __init__.py
│   ├── config.py               # dataclasses + YAML loading + inheritance; validation of invariants
│   │                           #   (e.g. d_inner % heads == 0, window % chunk == 0)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model.py            # embedding → blocks → head; owns nothing but composition
│   │   ├── block.py            # (mixer, ffn) unit with pre-norm residual, MoD wrapper
│   │   ├── mixers/
│   │   │   ├── base.py         # Mixer interface: forward(x, state, delta) -> (y, state)
│   │   │   ├── cfc.py          # CfC-mix (the layer under test)
│   │   │   ├── gated_conv.py   # LFM2-style baseline, same interface
│   │   │   ├── lru.py          # plain LRU baseline, same interface
│   │   │   └── attention.py    # SWA + chunk summaries; global flag for diagnostics
│   │   ├── ffn/
│   │   │   ├── dense.py        # SwiGLU
│   │   │   └── moe.py          # experts, router, bias-balancing, dropless dispatch
│   │   ├── mod.py              # Mixture-of-Depths: train router, causal predictor, gap counter
│   │   ├── mtp.py              # MTP head
│   │   ├── norm.py             # RMSNorm
│   │   ├── rope.py
│   │   ├── state.py            # SequenceState dataclass: per-layer h, conv_buf, kv ring,
│   │   │                       #   summaries, mod gaps; snapshot()/restore(); tensor-only, pickle-safe
│   │   └── init.py             # decay init (log-uniform a∈[0.9,0.999]), gate init, conv identity
│   │
│   ├── kernels/
│   │   ├── registry.py         # get("scan") -> reference or triton, chosen by env/config
│   │   ├── reference/
│   │   │   ├── scan.py         # sequential + chunked scan in plain torch
│   │   │   ├── cfc_step.py     # single-token step
│   │   │   ├── summary_mask.py # brute-force mask builder — the oracle
│   │   │   └── grouped_mm.py   # for-loop-over-experts MoE
│   │   └── triton/
│   │       ├── scan_fwd_bwd.py
│   │       ├── cfc_step_fused.py
│   │       ├── swa_summary_attn.py
│   │       ├── grouped_mm.py
│   │       └── int4_gemv.py
│   │
│   ├── data/
│   │   ├── stream.py           # uint16 memmap; order is a pure function of (seed, step)
│   │   ├── tokenizer.py        # wraps the 32k tokenizer; reserves <think>, </think>, effort tags now
│   │   ├── mixture.py          # domain weights, sampling
│   │   ├── packing.py          # sequence packing with chunk-aligned boundaries (C divides pack length)
│   │   └── loaders.py
│   │
│   ├── train/
│   │   ├── config.py           # TrainConfig: budget, LR schedule, resume fingerprint
│   │   ├── loop.py             # the driver: stepping, cadence, deadlines, resume. No torch.
│   │   ├── trainer.py          # torch adapter + CLI: `python -m deluge.train`
│   │   ├── losses.py           # ce + mtp + mod_pred + z-loss; weights from config
│   │   ├── schedule.py         # LR, MTP-weight anneal, length extension switch
│   │   ├── parallel.py         # FSDP / expert-parallel wrappers; the only file that imports them
│   │   ├── monitors.py         # decay histograms, MoD skip rate, router entropy, state norms
│   │   └── checkpoint.py       # atomic save/load incl. optimizer, scaler and RNG
│   │
│   ├── infer/
│   │   ├── engine.py           # decode loop; owns CUDA graphs and the state struct
│   │   ├── quant/
│   │   │   ├── export.py       # bf16 checkpoint -> int4/int8 packed weights
│   │   │   └── calibrate.py    # AWQ-style scale search
│   │   ├── graphs.py           # capture per-block graphs (process / skip variants)
│   │   ├── speculative.py      # MTP draft, batched verify, snapshot-per-position rollback
│   │   ├── cache.py            # KV ring + summary list, int8 key/value quant
│   │   ├── snapshots.py        # prefix cache store (disk + memory)
│   │   └── server.py           # minimal streaming API; optional
│   │
│   └── eval/
│       ├── perplexity.py
│       ├── tasks/              # lm-eval-harness adapters
│       ├── niah.py             # needle placement inside vs outside window, multi-needle
│       ├── speed.py            # tok/s decode & prefill, latency breakdown, cache bytes vs ctx
│       └── report.py           # writes docs/results/*.md — enforces "no speed number without baseline"
│
├── scripts/
│   ├── prepare_data.py         # corpus -> uint16 .bin + provenance .json
│   ├── export.py
│   ├── bench.py
│   ├── run_ladder.sh           # runs A0..A5 in order on the dev config
│   └── remote/                 # cluster launch, env, sync
│
├── tests/
│   ├── test_resume.py          # killing and restarting a run is bit-identical to not
│   ├── test_checkpoint.py      # atomicity, rotation, fallback past a corrupt newest
│   ├── test_data.py            # every sequence once per epoch; position is seek-able
│   ├── test_train_config.py    # budget invariants
│   ├── test_torch_backend.py   # the torch adapter; skipped without torch
│   ├── test_scan_parity.py     # triton == reference, fwd+bwd, random Δ
│   ├── test_mask_oracle.py     # attention with triton mask == brute-force oracle
│   ├── test_state_roundtrip.py # snapshot/restore bit-exact
│   ├── test_train_infer_equivalence.py  # teacher-forced logits (train path) == step-by-step (infer path)
│   ├── test_mod_causal.py      # inference never uses future info; predictor agrees with top-k on held-out
│   ├── test_speculative_golden.py       # greedy with speculation == greedy without
│   └── test_config_invariants.py
│
└── experiments/                # notebooks / one-off analyses; nothing here is imported by the package
```

## Notes on specific boundaries

- **`mixers/base.py` is the whole ablation strategy.** Every candidate cheap layer (CfC, conv, LRU) implements `forward(x, state, delta)`; swapping is a config string. A1 is a config change, not a code change.
- **`state.py` before `engine.py`.** Write the state struct and its round-trip test first; the engine, snapshots and speculation are all consumers of it.
- **`kernels/registry.py`** lets you develop the entire model with reference kernels on the 5070 Ti, then flip to Triton once parity tests pass. Never let a Triton kernel be the only implementation.
- **`data/packing.py`** must keep pack boundaries chunk-aligned (multiples of C) or the summary schedule drifts between train and inference.
- **`test_train_infer_equivalence.py`** is the most valuable test in the repo. It catches Δ-counter drift, mask drift, and RoPE drift in one shot. Run it on every commit that touches `model/` or `infer/`.
- **`train/loop.py` imports no torch, and that is the point.** Everything a resume depends on — data position, RNG, step accounting, when to save — lives there, so `test_resume.py` proves bit-identical restart in a tenth of a second on any machine. The torch adapter in `trainer.py` stays thin enough that its own tests only have to check that real modules and optimizers plug into the same sockets.
- **`docs/decisions/`** — one dated file per irreversible choice (e.g. "input-only gates", "summary from residual+state", "32k vocab"). Cheap now; invaluable when you write it up.

## Order of first commits

1. `config.py`, `model/` with reference kernels only, `state.py`, `test_state_roundtrip`, `test_config_invariants`.
2. `kernels/reference/summary_mask.py` + `attention.py` + `test_mask_oracle`.
3. `train/` minimal loop, dev config, A0 runs. (Loop, checkpointing, budgets and
   data ordering are done and tested; what they wait on is `deluge.model:build`.)
4. `infer/engine.py` in plain PyTorch + `test_train_infer_equivalence` — before any speed work.
5. Triton scan + parity test. Only now start caring about tok/s.
