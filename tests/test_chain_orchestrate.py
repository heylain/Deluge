"""orchestrate.tick against a fake Kaggle: I/O around step(), not step() itself."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from deluge.chain.kaggle import KaggleError, KaggleQuotaError
from deluge.chain.orchestrate import tick
from deluge.chain.step import MAX_API_ERRORS, initial_state

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]
QUEUE = """runs:
  - name: smoke
    model: configs/model/screen.yaml
    train: configs/train/smoke.yaml
    accelerator: cpu
    data: synthetic
"""


class FakeKaggle:
    def __init__(self, status="running", session=None, status_error=None, push_error=None):
        self._status, self._session = status, session
        self.status_error, self.push_error = status_error, push_error
        self.pushes, self.fetches = [], 0

    def status(self, slug):
        if self.status_error:
            raise self.status_error
        return self._status

    def fetch_session_json(self, slug):
        self.fetches += 1
        if isinstance(self._session, Exception):
            raise self._session
        return self._session

    def push(self, state, action):
        if self.push_error:
            raise self.push_error
        self.pushes.append((state["session"], action))


@pytest.fixture
def files(tmp_path):
    queue = tmp_path / "runs.yaml"
    queue.write_text(QUEUE)
    state = tmp_path / "state.json"
    state.write_text(json.dumps(initial_state(), indent=2) + "\n")
    return state, queue


def started(files, kaggle=None):
    state_path, queue = files
    tick(kaggle or FakeKaggle(), state_path, queue, NOW, "tick", "abc123")
    return json.loads(state_path.read_text())


def test_a_tick_with_nothing_to_do_writes_nothing(files):
    state_path, queue = files
    queue.write_text("runs: []\n")
    before = state_path.read_text()
    code, message = tick(FakeKaggle(), state_path, queue, NOW, "tick", "abc123")
    assert (code, message) == (0, None) and state_path.read_text() == before


def test_idle_tick_pushes_and_records(files):
    kaggle = FakeKaggle()
    state = started(files, kaggle)
    assert state["status"] == "running" and kaggle.pushes[0][1].first is True
    assert kaggle.pushes[0][0] == 1       # pushed with the new session number


def test_a_live_session_is_not_downloaded(files):
    state_path, queue = files
    started(files)
    kaggle = FakeKaggle(status="running")
    tick(kaggle, state_path, queue, NOW + timedelta(hours=1), "tick", "abc123")
    assert kaggle.fetches == 0


def test_api_errors_count_then_fail(files):
    state_path, queue = files
    started(files)
    kaggle = FakeKaggle(status_error=KaggleError("503"))
    codes = [tick(kaggle, state_path, queue, NOW + timedelta(hours=1), "tick", "x")[0]
             for _ in range(MAX_API_ERRORS)]
    assert codes == [0] * (MAX_API_ERRORS - 1) + [1]
    assert json.loads(state_path.read_text())["api_errors"] == MAX_API_ERRORS


def test_quota_refusal_on_push_waits(files):
    state = started(files, FakeKaggle(push_error=KaggleQuotaError("quota")))
    assert state["status"] == "waiting" and state["next_side"] == "a"


def test_a_failed_push_is_not_recorded_as_a_live_session(files):
    state = started(files, FakeKaggle(push_error=KaggleError("boom")))
    assert state["status"] == "idle" and state["api_errors"] == 1


def test_halting_writes_state_before_failing(files):
    state_path, queue = files
    state = started(files)
    later = NOW + timedelta(hours=1)
    for _ in range(2):
        crash = {"run_id": state["run"]["run_id"], "session": state["session"],
                 "exit_code": 1, "tokens_seen": 0}
        code, message = tick(FakeKaggle("complete", crash), state_path, queue,
                             later, "tick", "abc123")
        state = json.loads(state_path.read_text())
        later += timedelta(hours=1)
    assert code == 1 and state["status"] == "halted"
    assert "needs attention" in message


def test_committed_state_file_is_the_initial_state():
    committed = json.loads((ROOT / ".github/chain/state.json").read_text())
    assert committed == initial_state()


def test_an_unknown_status_is_not_downloaded_or_acted_on(files):
    state_path, queue = files
    before = started(files)
    kaggle = FakeKaggle(status="unknown")
    tick(kaggle, state_path, queue, NOW + timedelta(hours=1), "tick", "abc123")
    assert kaggle.fetches == 0 and kaggle.pushes == []
    assert json.loads(state_path.read_text())["session"] == before["session"]


def test_a_finished_kernel_whose_output_cannot_be_fetched_is_a_kill(files):
    # Review: a Kaggle-killed kernel may have no output at all; if fetching it
    # errors, that must read as "no session.json", not as an API outage that
    # never re-pushes.
    state_path, queue = files
    started(files)
    kaggle = FakeKaggle(status="error", session=KaggleError("no output"))
    tick(kaggle, state_path, queue, NOW + timedelta(hours=1), "tick", "abc123")
    state = json.loads(state_path.read_text())
    assert [a.side for _, a in kaggle.pushes] == ["a"]
    assert state["crash_streak"] == 1 and state["api_errors"] == 0


def test_a_broken_queue_does_not_stall_the_run_in_flight(files):
    # Review F6: the queue only matters when idle.
    state_path, queue = files
    state = started(files)
    queue.write_text("runs:\n  - name: Bad_Name\n")
    ok = {"run_id": state["run"]["run_id"], "session": state["session"],
          "exit_code": 2, "tokens_seen": 100}
    kaggle = FakeKaggle(status="complete", session=ok)
    code, _ = tick(kaggle, state_path, queue, NOW + timedelta(hours=1), "tick", "abc123")
    assert code == 0 and [a.side for _, a in kaggle.pushes] == ["b"]
