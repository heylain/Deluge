"""The chain's decision: given where it is and what Kaggle reports, what next.

Pure -- no I/O, no clock, no Kaggle -- so every row of the design's two
transition tables is a unit test (tests/test_chain_step.py). orchestrate.py
does the observing and acting around it.

The state is a plain dict because it is a JSON file in the repo
(.github/chain/state.json); initial_state() is its shape.
"""

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# Exit codes a session reports. 0 and 2 are deluge.train's (Outcome.exit_code).
# 3 and 4 are session.py's own: "no usable GPU", which is not the run's fault,
# and "failed before carrying the checkpoint forward" (clone, pip, no prior
# output), after which this side's output must never become a source.
DONE, RESUMABLE, NO_GPU, NO_HANDOFF = 0, 2, 3, 4

# A status read this soon after a push may still describe the previous version
# of the kernel -- and that version's session.json is right there to misread.
GRACE = timedelta(minutes=10)
# Kaggle kills a session at 12 h; the extra hour is queueing slack.
STUCK_AFTER = timedelta(hours=13)
# GPU quota is weekly; retrying sooner than this only burns API calls.
WAIT_FOR = timedelta(hours=6)
MAX_CRASHES = 2
MAX_API_ERRORS = 6                  # ~3 h of 30-minute ticks
# Quota resets weekly, so a week of no-GPU waits is not quota: the account
# cannot get a GPU at all (docs/kaggle-chain-spike.md, A8). Say so, once.
MAX_GPU_WAITS = 28                  # x WAIT_FOR = 7 days

State = Dict[str, Any]


@dataclass(frozen=True)
class Observation:
    status: str                     # queued|running|complete|error|cancelled|unknown
    session: Optional[Dict[str, Any]]   # the kernel's session.json, if any


@dataclass(frozen=True)
class Push:
    side: str
    first: bool                     # no kernel_sources: nothing has handed off yet


@dataclass(frozen=True)
class Fail:
    reason: str


Action = Union[None, Push, Fail]


def initial_state() -> State:
    return {"status": "idle", "run": None, "side": None, "session": 0,
            "pushed_at": None, "handed_off": False, "last_tokens": 0,
            "crash_streak": 0, "api_errors": 0, "gpu_waits": 0, "wait_until": None,
            "next_side": None, "done": []}


def kernel_slug(name: str, side: str) -> str:
    return f"deluge-{name}-{side}"


def other(side: str) -> str:
    return "b" if side == "a" else "a"


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


# ---- building blocks ------------------------------------------------------- #

def _push(s: State, side: str, now: datetime) -> Tuple[State, Action]:
    """Every push gets a fresh session number: (run_id, session) names it."""
    s.update(status="running", side=side, session=s["session"] + 1,
             pushed_at=_iso(now), wait_until=None, next_side=None)
    return s, Push(side=side, first=not s["handed_off"])


def _wait(s: State, side: str, now: datetime) -> Tuple[State, Action]:
    s.update(status="waiting", next_side=side, wait_until=_iso(now + WAIT_FOR))
    s["gpu_waits"] += 1
    if s["gpu_waits"] == MAX_GPU_WAITS:
        return s, Fail(f"{s['run']['name']}: no GPU for {MAX_GPU_WAITS} tries over "
                       f"{MAX_GPU_WAITS * WAIT_FOR} -- longer than a quota week. Is the "
                       f"Kaggle account phone-verified? The chain keeps retrying.")
    return s, None


def _failed(s: State, next_side: str, now: datetime, why: str) -> Tuple[State, Action]:
    s["crash_streak"] += 1
    if s["crash_streak"] >= MAX_CRASHES:
        s.update(status="halted", next_side=next_side)
        return s, Fail(f"{s['run']['name']}: {why} -- {s['crash_streak']} failures "
                       f"in a row without progress. Fix it, then run the chain "
                       f"workflow with command=resume (or skip).")
    return _push(s, next_side, now)


def _record(s: State, now: datetime, skipped: bool = False) -> Dict[str, Any]:
    record = {"name": s["run"]["name"], "run_id": s["run"]["run_id"],
              "commit": s["run"]["commit"], "sessions": s["session"],
              "finished": _iso(now)}
    if skipped:
        record["skipped"] = True
    return record


def _idle(s: State) -> State:
    done = s["done"]
    s.clear()
    s.update(initial_state(), done=done)
    return s


def _fresh(session: Optional[Dict[str, Any]], s: State) -> Optional[Dict[str, Any]]:
    """session.json only counts if it is this push's -- not an earlier one's."""
    if (isinstance(session, dict)
            and session.get("run_id") == s["run"]["run_id"]
            and session.get("session") == s["session"]):
        return session
    return None


# ---- the decision ------------------------------------------------------------ #

def step(state: State, queue: List[Dict[str, Any]], observation: Optional[Observation],
         now: datetime, command: str = "tick",
         commit: Optional[str] = None) -> Tuple[State, Action]:
    s = copy.deepcopy(state)
    s["api_errors"] = 0             # reaching step() means Kaggle answered, or wasn't asked

    if command == "skip":
        if s["run"] is None:
            return s, None
        s["done"].append(_record(s, now, skipped=True))
        return _idle(s), None
    if command == "resume":
        if s["status"] in ("halted", "waiting"):
            s["crash_streak"] = 0
            return _push(s, s["next_side"], now)
        return s, None

    status = s["status"]
    if status == "halted":
        return s, None
    if status == "waiting":
        if now >= datetime.fromisoformat(s["wait_until"]):
            return _push(s, s["next_side"], now)
        return s, None
    if status == "idle":
        return _start(s, queue, now, commit)
    if observation is None:
        raise ValueError("a running chain needs an observation of its kernel")
    return _running(s, observation, now)


def _start(s: State, queue, now, commit) -> Tuple[State, Action]:
    finished = {record["name"] for record in s["done"]}
    pending = [run for run in queue if run["name"] not in finished]
    if not pending:
        return s, None
    if not commit:
        raise ValueError("starting a run needs the commit to pin it to")
    run = pending[0]
    s.update(run={**run, "commit": commit, "run_id": f"{run['name']}@{_iso(now)}"},
             session=0, handed_off=False, last_tokens=0, crash_streak=0)
    return _push(s, "a", now)


def _running(s: State, observation: Observation, now: datetime) -> Tuple[State, Action]:
    pushed = datetime.fromisoformat(s["pushed_at"])
    if now - pushed < GRACE:
        return s, None
    side = s["side"]

    if observation.status in ("queued", "running"):
        if now - pushed > STUCK_AFTER:
            return _failed(s, side, now, f"kernel still {observation.status} after "
                                         f"{STUCK_AFTER}")
        return s, None

    session = _fresh(observation.session, s)
    if session is None:
        # Killed by Kaggle, or ended without writing session.json. Re-push the
        # same side: its input -- the other side's output -- still holds the
        # last good checkpoint, whatever this side's output did or did not keep.
        return _failed(s, side, now, f"kernel ended {observation.status} "
                                     f"without a session.json")

    code = session.get("exit_code")
    if code == NO_GPU:
        return _wait(s, side, now)
    s["gpu_waits"] = 0              # the session got the hardware it asked for
    if code == NO_HANDOFF:
        # This output holds no checkpoint; the other side's still does. Retry
        # here, and do not let this side become anyone's source.
        return _failed(s, side, now, "session failed before carrying the "
                                     "checkpoint forward (clone, pip or mount)")
    # A fresh session.json means this side's output exists and carries the
    # newest checkpoint (session.py copies it forward first), so hand off.
    s["handed_off"] = True
    tokens = session.get("tokens_seen") or 0
    progressed = tokens > s["last_tokens"]
    if progressed:
        s.update(crash_streak=0, last_tokens=tokens)
    if code == DONE:
        s["done"].append(_record(s, now))
        return _idle(s), None
    if code == RESUMABLE and progressed:
        return _push(s, other(side), now)
    why = ("session ended resumable but trained nothing" if code == RESUMABLE
           else f"training exited {code}")
    return _failed(s, other(side), now, why)


def api_error(state: State, reason: str) -> Tuple[State, Action]:
    """Kaggle did not answer. Change nothing else; fail once at the threshold."""
    s = copy.deepcopy(state)
    s["api_errors"] += 1
    if s["api_errors"] == MAX_API_ERRORS:
        return s, Fail(f"Kaggle API failing for {MAX_API_ERRORS} ticks in a row: {reason}")
    return s, None


def push_refused(state: State, now: datetime) -> Tuple[State, Action]:
    """Kaggle refused a push for quota: wait, then retry the same push."""
    return _wait(copy.deepcopy(state), state["side"], now)
