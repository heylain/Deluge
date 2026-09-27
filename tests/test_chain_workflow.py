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


def test_a_queued_tick_acts_on_the_latest_state_of_master():
    # Review F2: checkout defaults to the triggering SHA, so a tick queued
    # behind another would decide from stale state and push a second session.
    workflow, _ = load()
    job = workflow["jobs"]["tick"]
    assert job["if"] == "github.ref == 'refs/heads/master'"   # never commit state elsewhere
    checkout = job["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["ref"] == "master"
    assert checkout["with"]["fetch-depth"] == 0             # git show of pinned commits
    tick = next(s for s in job["steps"] if s.get("id") == "tick")
    assert '--commit "$(git rev-parse HEAD)"' in tick["run"]


def test_the_kaggle_cli_is_pinned_to_the_version_the_spike_checked():
    # Review F3: PUSH_OK and the status regex are 2.2.4's wording.
    workflow, _ = load()
    install = next(s for s in workflow["jobs"]["tick"]["steps"] if "pip install" in s.get("run", ""))
    assert '"kaggle==2.2.4"' in install["run"]


def test_a_state_push_that_races_a_human_push_is_retried():
    workflow, _ = load()
    commit = next(s for s in workflow["jobs"]["tick"]["steps"] if s.get("name") == "Commit state")
    assert "for attempt in" in commit["run"] and "exit 1" in commit["run"]
