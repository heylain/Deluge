"""The chain workflow's safety properties, read from the YAML itself."""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/chain.yml"


def load():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML reads the bare key `on` as boolean True.
    return workflow, workflow.get("on", workflow.get(True))


def test_runs_on_a_schedule_and_by_hand_only():
    _, triggers = load()
    assert set(triggers) == {"schedule", "workflow_dispatch"}   # no fork can reach the key
    assert triggers["schedule"] == [{"cron": "*/30 * * * *"}]
    command = triggers["workflow_dispatch"]["inputs"]["command"]
    assert command["options"] == ["tick", "resume", "skip"] and command["default"] == "tick"


def test_ticks_never_overlap_or_cancel_each_other():
    workflow, _ = load()
    assert workflow["concurrency"] == {"group": "chain", "cancel-in-progress": False}


def test_may_commit_the_state_file_and_nothing_else_is_granted():
    workflow, _ = load()
    assert workflow["permissions"] == {"contents": "write"}
    steps = workflow["jobs"]["tick"]["steps"]
    commit = next(s for s in steps if s.get("name") == "Commit state")
    assert "git add .github/chain/state.json" in commit["run"]
    assert "git pull --rebase" in commit["run"]
