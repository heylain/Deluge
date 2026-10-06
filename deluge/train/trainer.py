"""Torch backend and CLI for the training loop.

The loop itself is in loop.py and knows nothing about torch. This module is the
adapter: it builds the model, optimizer and loss scaler, and hands the loop a
`step_fn` that turns one step's micro-batches into a loss.

The model interface
-------------------
`--model-impl` names a factory, "module:attr", defaulting to deluge.model:build.

    build(model_config: ModelConfig) -> torch.nn.Module

and the module it returns is called as

    model(inputs, targets) -> loss | (loss, metrics)

with inputs and targets both int64 [batch, seq_len], and metrics a mapping of
name -> scalar tensor. The model owns its objective, not this file: spec 7's
loss is L_ce + 0.3 L_mtp + 0.01 L_mod_pred, and the MTP and MoD-predictor terms
are computed from tensors that never leave the model. Returning them in metrics
is what lets the run log show which term moved.

Two optional parts, used when present:
  - model(inputs, targets, "none") -> (token_losses [batch, seq_len], stats),
    for the held-out eval: reduction is the third positional argument, and the
    evaluator passes it positionally. stats maps names to tensors with a
    leading batch dimension (deluge/train/monitors.py reads decay_lambda,
    state_rms). Needed only when the data has a heldout.bin.
  - model.no_weight_decay() -> parameter names to leave undecayed beyond the
    1-D rule.

Data
----
`--data` is a uint16 token file. Beside it, optionally:
  - provenance.json: if it names a piece_len the data is read in pieces
    (deluge/data/stream.py), and it is copied into the run directory;
  - heldout.bin: the held-out set. With it the run evaluates every
    eval_every_tokens and once more when the budget is spent, and completes
    only once that final eval is written (final_eval.json).

Usage
-----
    python -m deluge.train \\
        --model configs/ablations/A0_scan_only.yaml \\
        --train configs/train/screen.yaml \\
        --data data/long4k/train.bin \\
        --out runs/A0

Exits 0 when the run is complete and 2 when the session ran out of time and the
run is resumable -- see docs/kaggle.md for the loop that reads that.
"""

import argparse
import contextlib
import importlib
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import ModelConfig, load_model_config
from ..data import (HELDOUT, PROVENANCE, TOKEN_DTYPE, TokenStream, load_provenance,
                    piece_batches)
from .checkpoint import Checkpointer, seed_everything
from .config import TrainConfig, load_train_config
from .loop import Outcome, TrainLoop, jsonl_logger
from .monitors import SUBSET_PIECES, EvalAccumulator, SkipCounter

TORCH_DTYPE = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def resolve_factory(spec: str) -> Callable[[ModelConfig], "torch.nn.Module"]:
    """Import a "module:attr" model factory, with an actionable error if absent."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"--model-impl must look like 'module:attr', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"cannot import {module_name!r} for --model-impl {spec!r}: {error}"
        ) from error
    try:
        return getattr(module, attr)
    except AttributeError as error:
        raise ImportError(
            f"{module_name!r} has no attribute {attr!r}; check --model-impl {spec!r}"
        ) from error


class _ScalerState:
    """Checkpoint adapter for GradScaler.

    A disabled scaler's state_dict() is empty, and loading an empty dict into an
    *enabled* one raises. Without this, a run smoke-tested on CPU could not be
    resumed on a GPU: the resume would throw, load_latest would fall back past
    every checkpoint, and the run would quietly restart from step 0.
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state):
        if state:
            self.scaler.load_state_dict(state)


def _autocast(cfg: TrainConfig, device: str):
    import torch
    if device.startswith("cuda") and cfg.dtype != "fp32":
        return torch.autocast(device_type="cuda", dtype=getattr(torch, TORCH_DTYPE[cfg.dtype]))
    return contextlib.nullcontext()


def build_backend(model_cfg: ModelConfig, cfg: TrainConfig, device: str,
                  data_parallel: bool, factory) -> Tuple[Callable, Dict]:
    """Construct model/optimizer/scaler and the step function that drives them."""
    import torch
    from torch.nn.utils import clip_grad_norm_

    # The initial weights follow cfg.seed (design 8.1: another seed changes the
    # init as well as the data order). TrainLoop.resume() seeds only once the
    # model exists, so without this they come from an unseeded generator.
    seed_everything(cfg.seed)
    model = factory(model_cfg).to(device)

    # Weight decay on matrices only. Norm gains and biases are 1-D and decaying
    # them pulls the normalisers toward zero, which is not what decay is for.
    # The model may name more: learned state, like the conv buffers and the
    # attention sinks, is 2-D or 3-D but no more a weight than a bias is.
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = set(getattr(model, "no_weight_decay", lambda: ())())
    unknown = sorted(no_decay - {n for n, _ in named})
    if unknown:
        raise ValueError(f"no_weight_decay() names {', '.join(unknown)}, which are "
                         f"not trainable parameters of the model")
    decayed = [p for n, p in named if p.ndim >= 2 and n not in no_decay]
    undecayed = [p for n, p in named if p.ndim < 2 or n in no_decay]
    optimizer = torch.optim.AdamW(
        [{"params": decayed, "weight_decay": cfg.weight_decay},
         {"params": undecayed, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
    )

    on_cuda = device.startswith("cuda")
    # fp16 needs a loss scaler; bf16's exponent range does not. The scaler
    # carries state (the current scale, and how long since it last overflowed),
    # so it is checkpointed alongside the optimizer -- resuming with a reset
    # scale replays the ramp and can spike the first steps after a restart.
    scaler = torch.amp.GradScaler(enabled=(cfg.dtype == "fp16" and on_cuda))
    skips = SkipCounter()
    autocast = _autocast(cfg, device)

    # DataParallel splits each micro-batch across Kaggle's two T4s. It is the
    # inefficient way to use two GPUs (~1.6x, against DDP's ~1.9x) but it is one
    # line and works inside a notebook, where torchrun does not. `model` stays
    # unwrapped in `stateful`, so checkpoints never carry a "module." prefix and
    # a two-GPU session resumes on one GPU unchanged.
    forward = torch.nn.DataParallel(model) if data_parallel else model

    def step_fn(batches: List[Tuple[np.ndarray, np.ndarray]], lr: float):
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        totals: Dict[str, float] = {}
        for inputs, targets in batches:
            x = torch.from_numpy(inputs).to(device, non_blocking=True)
            y = torch.from_numpy(targets).to(device, non_blocking=True)
            with autocast:
                output = forward(x, y)
            loss, metrics = output if isinstance(output, tuple) else (output, {})
            loss = loss.mean()          # DataParallel returns one loss per device
            scaler.scale(loss / len(batches)).backward()
            totals["loss"] = totals.get("loss", 0.0) + loss.item() / len(batches)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.mean()) / len(batches)

        if cfg.grad_clip > 0:
            # Unscale first: clipping scaled gradients would clip to the wrong
            # norm, by whatever factor the scaler happens to be at.
            scaler.unscale_(optimizer)
            totals["grad_norm"] = float(clip_grad_norm_(model.parameters(), cfg.grad_clip))
        scale = scaler.get_scale() if scaler.is_enabled() else None
        scaler.step(optimizer)
        scaler.update()
        if scale is not None:
            skips.observe(scale, scaler.get_scale())
            totals["loss_scale"] = scaler.get_scale()
        totals["skipped_steps"] = skips.count
        if on_cuda:
            # Peaks since the process started, max over the GPUs DataParallel
            # uses: what the M1 smoke sizes micro_batch by (docs/kaggle.md).
            gpus = range(torch.cuda.device_count())
            totals["peak_alloc_gib"] = round(
                max(torch.cuda.max_memory_allocated(i) for i in gpus) / 2**30, 3)
            totals["peak_reserved_gib"] = round(
                max(torch.cuda.max_memory_reserved(i) for i in gpus) / 2**30, 3)
        return totals

    return step_fn, {"model": model, "optimizer": optimizer,
                     "scaler": _ScalerState(scaler), "skips": skips}


def build_evaluator(model, cfg: TrainConfig, device: str, data_parallel: bool,
                    heldout: np.ndarray, piece_len: int) -> Callable:
    """The held-out eval: evaluate(n_pieces) -> (full, subset).

    Reads pieces 0 .. n_pieces - 1 of heldout.bin (None: all of them) in
    batches of micro_batch, under the training autocast, in eval mode with no
    grad. full covers every piece read; subset the first SUBSET_PIECES of them,
    or None if fewer were read. It touches neither the training RNG -- the
    model draws none -- nor the data position, and it leaves the model in the
    mode it found it in.
    """
    import torch

    forward = torch.nn.DataParallel(model) if data_parallel else model
    autocast = _autocast(cfg, device)

    def evaluate(n_pieces: Optional[int]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        full, subset, seen = EvalAccumulator(), EvalAccumulator(), 0
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for inputs, targets in piece_batches(heldout, piece_len, n_pieces,
                                                     cfg.micro_batch):
                    x = torch.from_numpy(inputs).to(device, non_blocking=True)
                    y = torch.from_numpy(targets).to(device, non_blocking=True)
                    with autocast:
                        # "none" positionally: DataParallel copies keyword
                        # arguments to every GPU, and a last batch of one
                        # piece would reach the second as keywords alone.
                        losses, stats = forward(x, y, "none")
                    losses = losses.float().cpu().numpy()
                    stats = {k: v.float().cpu().numpy() for k, v in stats.items()}
                    full.add(losses, stats)
                    keep = min(len(losses), SUBSET_PIECES - seen)
                    if keep > 0:
                        subset.add(losses[:keep], {k: v[:keep] for k, v in stats.items()})
                    seen += len(losses)
        finally:
            model.train(was_training)
        return full.result(), (subset.result() if subset.tokens else None)

    return evaluate


def in_run_eval(evaluate: Callable) -> Dict[str, Any]:
    """An "eval" record: the first SUBSET_PIECES pieces."""
    result, _ = evaluate(SUBSET_PIECES)
    return {"subset_loss": result["loss"], "subset_tokens": result["tokens"],
            "buckets": result["buckets"], "decay": result.get("decay"),
            "state_rms": result.get("state_rms")}


def final_eval(evaluate: Callable, skips: SkipCounter) -> Dict[str, Any]:
    """final_eval.json: the full set (the binding A1 metric), the subset (free:
    it is inside the full set), buckets and monitors over the full set, and the
    skip count design 7.3 bounds."""
    full, subset = evaluate(None)
    return {"full_loss": full["loss"], "full_tokens": full["tokens"],
            "subset_loss": subset["loss"], "subset_tokens": subset["tokens"],
            "buckets": full["buckets"], "decay": full.get("decay"),
            "state_rms": full.get("state_rms"), "skipped_steps": skips.count}



def run(args: argparse.Namespace) -> Outcome:
    model_cfg = load_model_config(args.model)
    cfg = load_train_config(args.train)
    if args.micro_batch:
        import dataclasses
        cfg = dataclasses.replace(cfg, micro_batch=args.micro_batch)
    if args.deadline_minutes is not None:
        import dataclasses
        cfg = dataclasses.replace(
            cfg, deadline_minutes=args.deadline_minutes or None)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = jsonl_logger(out / "log.jsonl")

    params = model_cfg.params()
    print(f"[train] {params.total/1e6:.1f}M params "
          f"({params.active_total/1e6:.1f}M active), "
          f"{cfg.total_steps} steps x {cfg.global_batch_tokens} tokens "
          f"= {cfg.total_tokens/1e9:.2f}B, "
          f"{cfg.grad_accum} x micro_batch {cfg.micro_batch}, dtype {cfg.dtype}")

    data = Path(args.data)
    provenance = load_provenance(data)
    # Before the model is built: a piece_len that does not fit seq_len stops
    # the run here, in a second, not after the model is on the GPU.
    stream = TokenStream.from_path(data, cfg.seq_len, cfg.seed)
    if provenance is not None:
        # Travels with the run: the exporter puts it in the HF run.json.
        shutil.copy2(data.parent / PROVENANCE, out / PROVENANCE)

    factory = resolve_factory(args.model_impl)
    step_fn, stateful = build_backend(
        model_cfg, cfg, args.device, args.data_parallel, factory)

    eval_fn = final_eval_fn = None
    heldout = data.parent / HELDOUT
    if heldout.exists():
        if not provenance or "piece_len" not in provenance:
            raise ValueError(f"{heldout} needs a {PROVENANCE} with piece_len beside "
                             f"it: the held-out set is read in pieces")
        evaluate = build_evaluator(
            stateful["model"], cfg, args.device, args.data_parallel,
            np.memmap(heldout, dtype=TOKEN_DTYPE, mode="r"), provenance["piece_len"])
        final_eval_fn = lambda: final_eval(evaluate, stateful["skips"])  # noqa: E731
        if cfg.eval_every_steps is not None:
            eval_fn = lambda: in_run_eval(evaluate)  # noqa: E731

    loop = TrainLoop(
        cfg=cfg,
        stream=stream,
        checkpointer=Checkpointer(out, keep_last=cfg.keep_last),
        stateful=stateful,
        step_fn=step_fn,
        meta={"train_fingerprint": cfg.fingerprint,
              "model_fingerprint": repr(model_cfg)},
        log_fn=log,
        eval_fn=eval_fn,
        final_eval_fn=final_eval_fn,
    )
    return loop.run(seed_dirs=[Path(d) for d in args.resume_from])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="model config YAML")
    parser.add_argument("--train", required=True, help="train config YAML")
    parser.add_argument("--data", required=True,
                        help="uint16 token .bin; provenance.json and heldout.bin "
                             "beside it are picked up")
    parser.add_argument("--out", required=True, help="checkpoint and log directory")
    parser.add_argument("--resume-from", action="append", default=[],
                        metavar="DIR",
                        help="extra read-only directory to look for checkpoints in; "
                             "on Kaggle, the previous session's output dataset")
    parser.add_argument("--model-impl", default="deluge.model:build",
                        help="model factory, 'module:attr'")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-parallel", action="store_true",
                        help="split each micro-batch over all visible GPUs "
                             "(Kaggle's 2xT4)")
    # Host-shaped overrides. Deliberately the only two: everything else that
    # affects the optimization lives in the config, where it is fingerprinted.
    parser.add_argument("--micro-batch", type=int, default=None,
                        help="override micro_batch to fit this GPU's VRAM")
    parser.add_argument("--deadline-minutes", type=float, default=None,
                        help="override the session deadline; 0 for none")

    args = parser.parse_args(argv)
    outcome = run(args)
    print(f"[train] {outcome.reason} at step {outcome.step} "
          f"({outcome.tokens_seen/1e9:.3f}B tokens), "
          f"checkpoint {outcome.checkpoint}")
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
