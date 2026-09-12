"""The training driver: stepping, checkpoint cadence, deadlines, resume.

Deliberately knows nothing about torch. Everything specific to the model lives
behind `step_fn`, which is handed the micro-batches for one optimizer step and
returns a loss. What is left is the part that has to be exactly right for a run
to survive being killed -- step accounting, data position, when to save, when to
stop -- and it is testable, in full, on a machine with no GPU.
"""

import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .checkpoint import Checkpointer, seed_everything
from .config import TrainConfig

Batch = Tuple[np.ndarray, np.ndarray]
# A step returns its loss, or a dict carrying "loss" plus any extra terms worth
# logging separately -- spec 7's objective is L_ce + 0.3 L_mtp + 0.01 L_mod_pred,
# and a run that only records the sum cannot tell which term moved.
StepFn = Callable[[List[Batch], float], Union[float, Dict[str, float]]]


class ResumeMismatch(RuntimeError):
    """A checkpoint was written under a config this run cannot continue."""


@dataclass(frozen=True)
class Outcome:
    step: int
    tokens_seen: int
    complete: bool
    reason: str                      # "complete" | "deadline" | "signal"
    checkpoint: Optional[Path]

    @property
    def exit_code(self) -> int:
        """0 when the budget is spent, 2 when the run is merely interrupted.

        A notebook reads this to decide whether to queue another session, so the
        two cases must not share an exit code.
        """
        return 0 if self.complete else 2


class TrainLoop:
    def __init__(
        self,
        cfg: TrainConfig,
        stream,
        checkpointer: Checkpointer,
        stateful: Dict[str, Any],
        step_fn: StepFn,
        meta: Optional[Dict[str, Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.cfg = cfg
        self.stream = stream
        self.checkpointer = checkpointer
        self.stateful = stateful
        self.step_fn = step_fn
        self.meta = dict(meta or {})
        self.clock = clock
        self.log_fn = log_fn or (lambda record: None)
        self.step = 0
        self._stop_reason: Optional[str] = None

    # ---- accounting ------------------------------------------------------ #

    @property
    def tokens_seen(self) -> int:
        """Derived from step, never stored: two counters would drift apart."""
        return self.step * self.cfg.global_batch_tokens

    def batches_for_step(self, step: int) -> List[Batch]:
        """Split one optimizer step's sequences into micro-batches, in order."""
        indices = self.stream.indices_for_step(step, self.cfg.sequences_per_step)
        size = self.cfg.micro_batch
        return [self.stream.batch(indices[i * size : (i + 1) * size])
                for i in range(self.cfg.grad_accum)]

    # ---- resume ---------------------------------------------------------- #

    def resume(self, seed_dirs: Iterable[Path] = ()) -> Optional[Path]:
        payload = self.checkpointer.load_latest(self.stateful, seed_dirs)
        if payload is None:
            seed_everything(self.cfg.seed)
            self.step = 0
            return None

        found = payload.get("meta", {})
        for key in ("train_fingerprint", "model_fingerprint"):
            if key in self.meta and key in found and found[key] != self.meta[key]:
                raise ResumeMismatch(
                    f"{payload['path']} was written with a different {key}. "
                    f"Continuing would apply this run's schedule to another "
                    f"run's weights; start a new output directory instead."
                )
        self.step = payload["step"]
        return payload["path"]

    # ---- stopping -------------------------------------------------------- #

    def _install_signal_handlers(self):
        def handle(signum, _frame):
            # Do not save here: the step is mid-backward and the optimizer state
            # is inconsistent. Record it and let the loop stop cleanly.
            self._stop_reason = f"signal:{signal.Signals(signum).name}"
        previous = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous[sig] = signal.signal(sig, handle)
            except ValueError:  # not on the main thread
                pass
        return previous

    @staticmethod
    def _restore_signal_handlers(previous):
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    # ---- run ------------------------------------------------------------- #

    def run(self, seed_dirs: Iterable[Path] = ()) -> Outcome:
        cfg = self.cfg
        resumed = self.resume(seed_dirs)
        self.log_fn({"event": "start", "step": self.step,
                     "total_steps": cfg.total_steps,
                     "resumed_from": str(resumed) if resumed else None})

        started = self.clock()
        deadline = (started + cfg.deadline_minutes * 60
                    if cfg.deadline_minutes is not None else None)
        previous_handlers = self._install_signal_handlers()

        last_saved = self.step
        last_step_seconds = 0.0
        reason = "complete"
        try:
            while self.step < cfg.total_steps:
                if self._stop_reason:
                    reason = self._stop_reason
                    break
                # Stop before a step that would overrun rather than after: an
                # interrupted step is wasted work, and on a 12 h host the margin
                # is worth more than one more step.
                if deadline is not None and self.clock() + last_step_seconds >= deadline:
                    reason = "deadline"
                    break

                began = self.clock()
                lr = cfg.lr_at(self.tokens_seen)
                result = self.step_fn(self.batches_for_step(self.step), lr)
                self.step += 1
                last_step_seconds = self.clock() - began

                if isinstance(result, dict):
                    if "loss" not in result:
                        raise KeyError(
                            "step_fn returned a dict without a 'loss' key; got "
                            f"{sorted(result)}")
                    metrics = dict(result)
                    loss = float(metrics.pop("loss"))
                else:
                    metrics, loss = {}, float(result)

                self.log_fn({
                    "event": "step", "step": self.step, "loss": loss, "lr": lr,
                    **metrics,
                    "tokens_seen": self.tokens_seen,
                    "seconds": round(last_step_seconds, 3),
                    "tokens_per_second": round(
                        cfg.global_batch_tokens / last_step_seconds, 1)
                    if last_step_seconds > 0 else None,
                })

                if self.step % cfg.checkpoint_every_steps == 0:
                    self._save()
                    last_saved = self.step
        finally:
            self._restore_signal_handlers(previous_handlers)

        # Always leave a checkpoint at the stopping point, or the session's
        # remaining work since the last cadence save is lost.
        path = self._save() if self.step != last_saved else self._latest_path()
        complete = self.step >= cfg.total_steps
        outcome = Outcome(step=self.step, tokens_seen=self.tokens_seen,
                          complete=complete,
                          reason="complete" if complete else reason,
                          checkpoint=path)
        self.log_fn({"event": "stop", "step": outcome.step,
                     "reason": outcome.reason, "complete": outcome.complete,
                     "checkpoint": str(path) if path else None,
                     "elapsed_seconds": round(self.clock() - started, 1)})
        return outcome

    def _save(self) -> Path:
        return self.checkpointer.save(
            self.step, self.stateful,
            {**self.meta, "tokens_seen": self.tokens_seen})

    def _latest_path(self) -> Optional[Path]:
        found = self.checkpointer.candidates()
        return found[0][1] if found else None


def jsonl_logger(path: Path, echo: bool = True) -> Callable[[Dict[str, Any]], None]:
    """Append-mode run log. Appends, so a resumed session extends one file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(record: Dict[str, Any]) -> None:
        record = {"t": round(time.time(), 3), **record}
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        if echo and record.get("event") != "step":
            print(f"[train] {record}")
        elif echo and record["step"] % 10 == 0:
            print(f"[train] step {record['step']} loss {record['loss']:.4f} "
                  f"lr {record['lr']:.2e} {record['tokens_per_second']} tok/s")

    return log
