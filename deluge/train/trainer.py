"""Torch backend and CLI for the training loop.

The loop itself is in loop.py and knows nothing about torch. This module is the
adapter: it builds the model, optimizer and loss scaler, and hands the loop a
`step_fn` that turns one step's micro-batches into a loss.

The interface M1 has to satisfy
------------------------------
`--model-impl` names a factory, "module:attr", defaulting to deluge.model:build.

    build(model_config: ModelConfig) -> torch.nn.Module

and the module it returns is called as

    model(inputs, targets) -> loss | (loss, metrics)

with inputs and targets both int64 [batch, seq_len], and metrics a mapping of
name -> scalar tensor. The model owns its objective, not this file: spec 7's
loss is L_ce + 0.3 L_mtp + 0.01 L_mod_pred, and the MTP and MoD-predictor terms
are computed from tensors that never leave the model. Returning them in metrics
is what lets the run log show which term moved.

Usage
-----
    python -m deluge.train \\
        --model configs/model/dev.yaml \\
        --train configs/train/screen.yaml \\
        --data data/tokens.bin \\
        --out runs/A0

Exits 0 when the budget is spent and 2 when the session ran out of time and the
run is resumable -- see docs/kaggle.md for the loop that reads that.
"""

import argparse
import contextlib
import importlib
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..config import ModelConfig, load_model_config
from ..data import TokenStream
from .checkpoint import Checkpointer
from .config import TrainConfig, load_train_config
from .loop import Outcome, TrainLoop, jsonl_logger

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
            f"cannot import {module_name!r} for --model-impl {spec!r}: {error}. "
            f"The model is M1; until it lands, point --model-impl at a stub."
        ) from error
    try:
        return getattr(module, attr)
    except AttributeError as error:
        # deluge.model exists from M1's first commit, well before build() does.
        raise ImportError(
            f"{module_name!r} has no attribute {attr!r}. "
            f"The model is M1; until it lands, point --model-impl at a stub."
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


def build_backend(model_cfg: ModelConfig, cfg: TrainConfig, device: str,
                  data_parallel: bool, factory) -> Tuple[Callable, Dict]:
    """Construct model/optimizer/scaler and the step function that drives them."""
    import torch
    from torch.nn.utils import clip_grad_norm_

    model = factory(model_cfg).to(device)

    # Weight decay on matrices only. Norm gains and biases are 1-D and decaying
    # them pulls the normalisers toward zero, which is not what decay is for.
    decayed = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    undecayed = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
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
    autocast = (
        torch.autocast(device_type="cuda", dtype=getattr(torch, TORCH_DTYPE[cfg.dtype]))
        if on_cuda and cfg.dtype != "fp32" else contextlib.nullcontext()
    )

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
        scaler.step(optimizer)
        scaler.update()
        return totals

    return step_fn, {"model": model, "optimizer": optimizer,
                     "scaler": _ScalerState(scaler)}



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

    factory = resolve_factory(args.model_impl)
    step_fn, stateful = build_backend(
        model_cfg, cfg, args.device, args.data_parallel, factory)

    loop = TrainLoop(
        cfg=cfg,
        stream=TokenStream.from_path(args.data, cfg.seq_len, cfg.seed),
        checkpointer=Checkpointer(out, keep_last=cfg.keep_last),
        stateful=stateful,
        step_fn=step_fn,
        meta={"train_fingerprint": cfg.fingerprint,
              "model_fingerprint": repr(model_cfg)},
        log_fn=log,
    )
    return loop.run(seed_dirs=[Path(d) for d in args.resume_from])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="model config YAML")
    parser.add_argument("--train", required=True, help="train config YAML")
    parser.add_argument("--data", required=True, help="uint16 token .bin")
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
