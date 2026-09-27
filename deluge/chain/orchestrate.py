"""One tick of the Kaggle chain: observe, decide, act, record.

Run by .github/workflows/chain.yml every 30 minutes. The decision is
deluge.chain.step's; this module only does the I/O around it. It writes the
state file only when the state changed, so that file's git history reads as
the chain's log, and a failed push is never recorded as a live session.

Exit status 1 means the chain needs a human (halted, or Kaggle unreachable
for ~3 h): the failed workflow run is what emails you. Otherwise 0.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .kaggle import KaggleCLI, KaggleError, KaggleQuotaError
from .queue import load_queue
from .step import (MAX_API_ERRORS, Fail, Observation, Push, api_error, kernel_slug,
                   push_refused, step)

STATE = Path(".github/chain/state.json")
QUEUE = Path("configs/chain/runs.yaml")
# Only a finished kernel has a session.json worth downloading.
FINISHED = ("complete", "error", "cancelled")


def observe(kaggle, state: dict) -> Observation:
    slug = kernel_slug(state["run"]["name"], state["side"])
    status = kaggle.status(slug)
    session = None
    if status in FINISHED:
        try:
            session = kaggle.fetch_session_json(slug)
        except KaggleError:
            # Kaggle answered the status; a killed kernel may simply have no
            # output. That is "no session.json" -- a kill -- not an outage.
            session = None
    return Observation(status=status, session=session)


def tick(kaggle, state_path: Path, queue_path: Path, now: datetime, command: str,
         commit: Optional[str]) -> Tuple[int, Optional[str]]:
    state = json.loads(state_path.read_text())
    # The queue only matters when idle; a typo in a future entry must not
    # stall the run in flight.
    queue = load_queue(queue_path) if state["status"] == "idle" and command == "tick" else []
    try:
        needs_look = state["status"] == "running" and command == "tick"
        observation = observe(kaggle, state) if needs_look else None
    except KaggleError as error:
        new, action = api_error(state, str(error))
    else:
        new, action = step(state, queue, observation, now, command, commit)
        if isinstance(action, Push):
            try:
                kaggle.push(new, action)
            except KaggleQuotaError:
                new, action = push_refused(new, now)
            except KaggleError as error:
                # Keep the old state: the push did not happen, and the next
                # tick will make the same decision and try it again.
                new, action = api_error(state, str(error))

    message = None
    if new != state:
        state_path.write_text(json.dumps(new, indent=2) + "\n")
        message = describe(state, new, action)
    if isinstance(action, Fail):
        print(f"[chain] needs attention: {action.reason}", file=sys.stderr)
        return 1, message
    return 0, message


def describe(old: dict, new: dict, action) -> str:
    name = (new["run"] or old["run"] or {}).get("name", "-")
    if isinstance(action, Fail):
        return f"chain: {name} needs attention -- {action.reason}"
    if isinstance(action, Push):
        return f"chain: {name} session {new['session']} pushed on {action.side}"
    if len(new["done"]) > len(old["done"]):
        last = new["done"][-1]
        verb = "skipped" if last.get("skipped") else "done"
        return f"chain: {last['name']} {verb} after {last['sessions']} session(s)"
    if new["status"] == "waiting":
        return f"chain: {name} waiting for GPU quota until {new['wait_until']}"
    if new["api_errors"] > old["api_errors"]:
        return f"chain: Kaggle API error {new['api_errors']}/{MAX_API_ERRORS}"
    return "chain: state updated"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--command", choices=("tick", "resume", "skip"), default="tick")
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA"),
                        help="commit a newly started run is pinned to")
    parser.add_argument("--message-file", type=Path,
                        help="written with the commit message when the state changed")
    args = parser.parse_args(argv)

    owner = os.environ.get("KAGGLE_USERNAME")
    if not owner:
        parser.error("KAGGLE_USERNAME is not set (a repo secret, in the workflow)")
    code, message = tick(KaggleCLI(owner), args.state, args.queue,
                         datetime.now(timezone.utc), args.command, args.commit)
    if message:
        print(message)
        if args.message_file:
            args.message_file.write_text(message + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
