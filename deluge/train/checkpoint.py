"""Atomic, resumable checkpoints.

Written for a host that will kill the process without warning. Kaggle stops a
session at 12 h; a spot instance stops whenever. Two things follow:

  - a half-written checkpoint must never be mistaken for a whole one, so the
    payload is written to a temporary name and renamed into place only once it
    is on disk. os.replace is atomic, so the final name either does not exist
    or is complete;
  - the newest checkpoint is not trusted blindly. If it fails to deserialize,
    load_latest falls back to the one before it rather than failing the run.

Everything needed to continue is in the payload, RNG included. "Resume" means
the next step is bit-identical to the step the killed process would have run,
not merely similar -- tests/test_resume.py asserts exactly that.

torch is optional here: the container, the rotation and the RNG plumbing for
python and numpy work without it, which is what lets the resume equivalence be
tested on a machine with no GPU and no torch.
"""

import os
import pickle
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol

import numpy as np

try:  # torch is a [train] extra, not a base dependency
    import torch
except ImportError:  # pragma: no cover - exercised by the torch-free test run
    torch = None

STEP_RE = re.compile(r"^step-(\d{9})\.pt$")
TMP_PREFIX = "tmp-checkpoint-"


class Stateful(Protocol):
    """What a checkpointed object must provide: torch modules and optimizers do."""

    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state: Dict[str, Any]) -> Any: ...


# --------------------------------------------------------------------------- #
# RNG
# --------------------------------------------------------------------------- #

def rng_state() -> Dict[str, Any]:
    """Capture every generator the training step can draw from."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore generators, skipping any the current host cannot provide.

    A checkpoint written on 2 GPUs and resumed on 1 restores the CPU generators
    and leaves CUDA to be reseeded; dropout masks differ from that point, which
    is why the resume test asserts equivalence on a fixed device count.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if torch is None:
        return
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    cuda = state.get("cuda")
    if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #

def _dump(payload: Dict[str, Any], path: Path) -> None:
    """Write payload to path, durably: flushed and fsynced before it is named."""
    fd, tmp_name = tempfile.mkstemp(prefix=TMP_PREFIX, dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            if torch is not None:
                torch.save(payload, handle)
            else:
                pickle.dump(payload, handle, protocol=4)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        # fsync the directory too, or the rename itself can be lost on power cut
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _load(path: Path) -> Dict[str, Any]:
    if torch is not None:
        # weights_only=False: the payload is our own, and carries the step
        # counter and RNG state as plain objects, not just tensors.
        return torch.load(path, map_location="cpu", weights_only=False)
    with open(path, "rb") as handle:
        return pickle.load(handle)


# --------------------------------------------------------------------------- #
# Checkpointer
# --------------------------------------------------------------------------- #

class Checkpointer:
    """Saves into out_dir; resumes from out_dir plus any read-only seed dirs.

    The seed dirs are how a Kaggle run continues: the previous session's output
    is attached as a read-only input dataset, so the newest checkpoint may live
    somewhere this process cannot write.
    """

    def __init__(self, out_dir: Path, keep_last: int = 2):
        if keep_last < 1:
            raise ValueError("keep_last must be at least 1")
        self.out_dir = Path(out_dir)
        self.keep_last = keep_last
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- discovery ------------------------------------------------------- #

    @staticmethod
    def _checkpoints_in(directory: Path) -> list[tuple[int, Path]]:
        if not directory.is_dir():
            return []
        found = []
        for entry in directory.iterdir():
            match = STEP_RE.match(entry.name)
            if match:
                found.append((int(match.group(1)), entry))
        return sorted(found)

    def candidates(self, search_dirs: Iterable[Path] = ()) -> list[tuple[int, Path]]:
        """Every checkpoint visible to this run, newest first."""
        found: list[tuple[int, Path]] = []
        for directory in [self.out_dir, *(Path(d) for d in search_dirs)]:
            found.extend(self._checkpoints_in(directory))
        # Newest first. sorted() is stable, so equal steps keep discovery order
        # and out_dir (searched first) wins: a resumed session prefers its own
        # output over the read-only snapshot it started from.
        return sorted(found, key=lambda pair: pair[0], reverse=True)

    def newest_readable(self, search_dirs: Iterable[Path] = ()) -> Optional[tuple[int, Path]]:
        """The newest checkpoint that deserializes, without restoring it.

        What a new session copies forward (deluge/chain/session.py). It uses the
        same discovery and the same loader as load_latest, so "the checkpoint the
        next session will resume from" cannot drift between the two.
        """
        for step, path in self.candidates(search_dirs):
            try:
                _load(path)
            except Exception:  # noqa: BLE001 - any failure means "try older"
                continue
            return step, path
        return None

    # ---- save ------------------------------------------------------------ #

    def save(self, step: int, stateful: Dict[str, Stateful],
             meta: Optional[Dict[str, Any]] = None) -> Path:
        payload = {
            "step": step,
            "meta": dict(meta or {}),
            "rng": rng_state(),
            "state": {name: obj.state_dict() for name, obj in stateful.items()},
        }
        path = self.out_dir / f"step-{step:09d}.pt"
        _dump(payload, path)
        self._rotate()
        return path

    def _rotate(self) -> None:
        """Keep the newest keep_last checkpoints, and sweep abandoned temps."""
        existing = self._checkpoints_in(self.out_dir)
        for _, path in existing[: max(0, len(existing) - self.keep_last)]:
            path.unlink(missing_ok=True)
        for entry in self.out_dir.iterdir():
            if entry.name.startswith(TMP_PREFIX):
                entry.unlink(missing_ok=True)

    # ---- load ------------------------------------------------------------ #

    def load_latest(self, stateful: Dict[str, Stateful],
                    search_dirs: Iterable[Path] = ()) -> Optional[Dict[str, Any]]:
        """Restore the newest readable checkpoint. Returns its payload, or None.

        Objects are mutated in place. A checkpoint that fails to deserialize --
        or that does not carry every object this run expects -- is skipped in
        favour of an older one, because a run that silently starts from step 0
        after 11 h of training is worse than one that starts from step N-1.
        """
        for step, path in self.candidates(search_dirs):
            try:
                payload = _load(path)
                missing = sorted(set(stateful) - set(payload["state"]))
                if missing:
                    raise KeyError(f"checkpoint lacks state for {', '.join(missing)}")
                for name, obj in stateful.items():
                    obj.load_state_dict(payload["state"][name])
                set_rng_state(payload["rng"])
            except Exception as error:  # noqa: BLE001 - any failure means "try older"
                print(f"[checkpoint] ignoring {path} at step {step}: {error!r}")
                continue
            payload["path"] = path
            return payload
        return None
