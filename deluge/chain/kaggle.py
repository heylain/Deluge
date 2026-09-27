"""The kaggle CLI, wrapped: status, fetch session.json, push a session.

Thin on purpose. Everything that decides lives in step.py; this module turns
CLI output into step.py's vocabulary and raises one error type for "Kaggle did
not do it", with quota refusals split out because the chain waits on those
instead of counting them. Constants here were checked against the real CLI in
docs/kaggle-chain-spike.md.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from .session import RUN_MARKER
from .step import Push, kernel_slug, other

SESSION_PY = Path(__file__).with_name("session.py")
REPO = "https://github.com/heylain/Deluge"
MACHINE_SHAPE = "NvidiaTeslaT4"     # the P100 cannot run Triton (sm_60)
PUSH_OK = "successfully pushed"
STATUS_RE = re.compile(r'status "([^"]+)"')
STATUS_ALIASES = {
    "queued": "queued", "running": "running", "complete": "complete",
    "error": "error", "cancelrequested": "cancelled",
    "cancelacknowledged": "cancelled", "cancelled": "cancelled",
}


class KaggleError(RuntimeError):
    """Kaggle did not do what was asked. The chain retries on the next tick."""


class KaggleQuotaError(KaggleError):
    """Kaggle refused for quota. The chain waits rather than counting a failure."""


def _raise_for(text: str) -> None:
    error = KaggleQuotaError if "quota" in text.lower() else KaggleError
    raise error(text.strip() or "kaggle CLI failed with no output")


def render(dest: Path, owner: str, state: dict, action: Push,
           session_source: Path = SESSION_PY) -> None:
    """Write the kernel directory `kaggle kernels push -p` expects."""
    run = state["run"]
    source = session_source.read_text()
    marker = re.compile(rf"^{re.escape(RUN_MARKER)}.*$", re.MULTILINE)
    if not marker.search(source):
        raise KaggleError(f"{session_source} has no line starting {RUN_MARKER!r} to render")
    params = {"run": run, "session": state["session"], "side": action.side,
              "first": action.first, "repo": REPO}
    # json inside a Python string literal: JSON's true/null are not Python.
    rendered = marker.sub(
        lambda _: f"RUN = json.loads({json.dumps(json.dumps(params))})", source, count=1)

    gpu = run["accelerator"] == "gpu"
    slug = kernel_slug(run["name"], action.side)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "session.py").write_text(rendered)
    (dest / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": "session.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": True,        # the session clones the repo
        "enable_gpu": gpu,
        "machine_shape": MACHINE_SHAPE if gpu else "",
        "dataset_sources": [] if run["data"] == "synthetic" else [run["data"].split(":")[0]],
        "competition_sources": [],
        "kernel_sources": [] if action.first
                          else [f"{owner}/{kernel_slug(run['name'], other(action.side))}"],
    }, indent=2) + "\n")


class KaggleCLI:
    def __init__(self, owner: str, runner: Callable = subprocess.run, binary: str = "kaggle"):
        self.owner, self.runner, self.binary = owner, runner, binary

    def _run(self, *args: str) -> str:
        result = self.runner([self.binary, *args], capture_output=True, text=True)
        if result.returncode != 0:
            _raise_for(f"{result.stderr}\n{result.stdout}")
        return result.stdout

    def status(self, slug: str) -> str:
        out = self._run("kernels", "status", f"{self.owner}/{slug}")
        match = STATUS_RE.search(out)
        if not match:
            raise KaggleError(f"cannot read a status from {out!r}")
        # "KernelWorkerStatus.CANCEL_REQUESTED" and "cancelRequested" alike
        raw = match.group(1).rsplit(".", 1)[-1].replace("_", "").lower()
        return STATUS_ALIASES.get(raw, "unknown")

    def fetch_session_json(self, slug: str) -> Optional[dict]:
        with tempfile.TemporaryDirectory() as tmp:
            self._run("kernels", "output", f"{self.owner}/{slug}", "-p", tmp,
                      "--file-pattern", r"^session\.json$")
            path = Path(tmp) / "session.json"
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return None

    def push(self, state: dict, action: Push) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            render(Path(tmp), self.owner, state, action)
            out = self._run("kernels", "push", "-p", tmp)
        if PUSH_OK not in out.lower():
            _raise_for(out)
