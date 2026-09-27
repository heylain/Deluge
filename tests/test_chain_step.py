"""Every transition in the chain design's two tables, one test each.

step() is pure, so these need no Kaggle, no clock and no GitHub.
"""

from datetime import datetime, timedelta, timezone

import pytest

from deluge.chain.step import (
    GRACE, MAX_API_ERRORS, MAX_GPU_WAITS, STUCK_AFTER, WAIT_FOR, Fail, Observation, Push,
    api_error, initial_state, kernel_slug, push_refused, step,
)

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
RUN = {"name": "smoke", "model": "m.yaml", "train": "t.yaml",
       "model_impl": "deluge.train.smoke:build", "accelerator": "cpu",
       "data": "synthetic"}


def iso(t):
    return t.isoformat(timespec="seconds")


def running(side="a", session=1, handed_off=False, pushed=NOW - timedelta(hours=2),
            **overrides):
    state = initial_state()
    state.update(status="running", side=side, session=session,
                 handed_off=handed_off, pushed_at=iso(pushed),
                 run={**RUN, "commit": "abc123", "run_id": "smoke@2026-09-27T00:00:00+00:00"})
    state.update(overrides)
    return state


def report(state, exit_code, tokens, status="complete", **overrides):
    session = {"run_id": state["run"]["run_id"], "session": state["session"],
               "side": state["side"], "exit_code": exit_code,
               "tokens_seen": tokens, "commit": "abc123", **overrides}
    return Observation(status=status, session=session)


# ---- starting ------------------------------------------------------------ #

def test_idle_starts_the_first_queued_run_on_a_with_no_sources():
    state, action = step(initial_state(), [RUN], None, NOW, commit="abc123")
    assert action == Push(side="a", first=True)
    assert state["status"] == "running" and state["session"] == 1
    assert state["run"]["commit"] == "abc123"
    assert state["run"]["run_id"] == f"smoke@{iso(NOW)}"
    assert state["pushed_at"] == iso(NOW)


def test_idle_skips_runs_already_done():
    start = initial_state()
    start["done"] = [{"name": "smoke"}]
    later = {**RUN, "name": "a0"}
    state, action = step(start, [RUN, later], None, NOW, commit="abc123")
    assert action == Push(side="a", first=True) and state["run"]["name"] == "a0"


def test_idle_with_nothing_queued_does_nothing():
    start = initial_state()
    start["done"] = [{"name": "smoke"}]
    state, action = step(start, [RUN], None, NOW, commit="abc123")
    assert action is None and state == start


def test_starting_a_run_needs_a_commit_to_pin():
    with pytest.raises(ValueError, match="commit"):
        step(initial_state(), [RUN], None, NOW)


def test_step_does_not_mutate_its_input():
    start = running()
    before = repr(start)
    step(start, [RUN], report(start, 2, 100), NOW)
    assert repr(start) == before


# ---- the normal path ------------------------------------------------------ #

def test_no_decision_within_grace_after_push():
    # Review focus 1: the status may still be the previous version's.
    start = running(pushed=NOW - GRACE + timedelta(seconds=1))
    state, action = step(start, [RUN], report(start, 2, 100), NOW)
    assert action is None and state == start


@pytest.mark.parametrize("status", ["queued", "running"])
def test_a_live_session_is_left_alone(status):
    start = running()
    state, action = step(start, [RUN], Observation(status, None), NOW)
    assert action is None and state == start


def test_resumable_exit_hands_off_to_the_other_side():
    start = running(crash_streak=1)
    state, action = step(start, [RUN], report(start, 2, 5000), NOW)
    assert action == Push(side="b", first=False)
    assert state["side"] == "b" and state["session"] == 2
    assert state["last_tokens"] == 5000 and state["crash_streak"] == 0
    assert state["handed_off"] is True


def test_completion_records_the_run_and_goes_idle():
    start = running(side="a", session=3, handed_off=True, last_tokens=900)
    state, action = step(start, [RUN], report(start, 0, 1000), NOW)
    assert action is None and state["status"] == "idle" and state["run"] is None
    [done] = state["done"]
    assert done["name"] == "smoke" and done["sessions"] == 3
    assert done["commit"] == "abc123" and done["finished"] == iso(NOW)


# ---- failures -------------------------------------------------------------- #

def test_training_crash_hands_off_and_counts():
    start = running(last_tokens=100)
    state, action = step(start, [RUN], report(start, 1, 100), NOW)
    assert action == Push(side="b", first=False)
    assert state["crash_streak"] == 1


def test_second_crash_without_progress_halts():
    start = running(side="b", handed_off=True, crash_streak=1, last_tokens=100)
    state, action = step(start, [RUN], report(start, 1, 100), NOW)
    assert isinstance(action, Fail) and "smoke" in action.reason
    assert state["status"] == "halted" and state["next_side"] == "a"


def test_a_crash_after_progress_starts_the_count_again():
    start = running(crash_streak=1, last_tokens=100)
    state, action = step(start, [RUN], report(start, 1, 200), NOW)
    assert isinstance(action, Push) and state["crash_streak"] == 1


def test_exit_2_without_progress_counts_as_a_failure():
    # Review focus 2: a session that trains nothing must not loop forever.
    start = running(last_tokens=100)
    state, action = step(start, [RUN], report(start, 2, 100), NOW)
    assert action == Push(side="b", first=False) and state["crash_streak"] == 1


@pytest.mark.parametrize("status", ["error", "cancelled", "unknown", "complete"])
def test_kaggle_kill_without_session_json_retries_the_same_side(status):
    start = running(side="a", session=1, handed_off=False)
    state, action = step(start, [RUN], Observation(status, None), NOW)
    # Same side: its input is unchanged. Still first: nothing has handed off.
    assert action == Push(side="a", first=True)
    assert state["session"] == 2 and state["crash_streak"] == 1


def test_a_stale_session_json_is_ignored():
    start = running(session=4)
    stale = report(start, 2, 100, session=3)
    state, action = step(start, [RUN], stale, NOW)
    assert action == Push(side="a", first=True) and state["crash_streak"] == 1


def test_a_session_json_from_an_earlier_run_is_ignored():
    start = running()
    stale = report(start, 0, 100, run_id="smoke@2026-01-01T00:00:00+00:00")
    _, action = step(start, [RUN], stale, NOW)
    assert action == Push(side="a", first=True)


def test_stuck_session_is_treated_as_a_kill():
    start = running(pushed=NOW - STUCK_AFTER - timedelta(minutes=1))
    state, action = step(start, [RUN], Observation("running", None), NOW)
    assert action == Push(side="a", first=True) and state["crash_streak"] == 1


def test_no_gpu_waits_without_counting_a_crash():
    start = running(side="b", handed_off=True)
    state, action = step(start, [RUN], report(start, 3, 0), NOW)
    assert action is None and state["status"] == "waiting"
    assert state["wait_until"] == iso(NOW + WAIT_FOR)
    assert state["next_side"] == "b" and state["crash_streak"] == 0


def test_waiting_until_the_backoff_ends_then_pushing():
    start = running(side="b", handed_off=True)
    start.update(status="waiting", next_side="b", wait_until=iso(NOW + timedelta(hours=1)))
    state, action = step(start, [RUN], None, NOW)
    assert action is None and state == start
    state, action = step(start, [RUN], None, NOW + timedelta(hours=1))
    assert action == Push(side="b", first=False) and state["status"] == "running"
    assert state["wait_until"] is None and state["next_side"] is None


def test_push_refused_for_quota_waits_on_the_side_it_tried():
    start = running(side="b", session=2, handed_off=True)
    state, action = push_refused(start, NOW)
    assert action is None
    assert state["status"] == "waiting" and state["next_side"] == "b"
    assert state["wait_until"] == iso(NOW + WAIT_FOR)


def test_a_week_without_a_gpu_fails_once():
    # Quota resets weekly, so a longer drought is not quota: an account that
    # cannot get a GPU at all (spike A8) must not wait forever in silence.
    state, actions = running(), []
    for _ in range(MAX_GPU_WAITS + 1):
        state, action = push_refused(state, NOW)
        actions.append(action)
    assert MAX_GPU_WAITS * WAIT_FOR >= timedelta(days=7)
    assert actions[:MAX_GPU_WAITS - 1] == [None] * (MAX_GPU_WAITS - 1)
    assert isinstance(actions[MAX_GPU_WAITS - 1], Fail)
    assert actions[MAX_GPU_WAITS] is None
    assert state["status"] == "waiting"          # still retrying, just not silently


def test_no_gpu_reports_count_toward_the_drought():
    start = running(gpu_waits=MAX_GPU_WAITS - 1)
    state, action = step(start, [RUN], report(start, 3, 0), NOW)
    assert isinstance(action, Fail) and "GPU" in action.reason
    assert state["status"] == "waiting"


def test_a_session_that_ran_clears_the_gpu_drought():
    start = running(gpu_waits=5)
    state, _ = step(start, [RUN], report(start, 2, 100), NOW)
    assert state["gpu_waits"] == 0


# ---- halted, commands, API errors ------------------------------------------- #

def test_a_halted_chain_stays_quiet_on_tick():
    start = running(status="halted", next_side="a", crash_streak=2)
    state, action = step(start, [RUN], None, NOW)
    assert action is None and state == start


def test_resume_restarts_a_halted_chain_on_the_side_it_would_have_pushed():
    start = running(status="halted", next_side="b", crash_streak=2, handed_off=True)
    state, action = step(start, [RUN], None, NOW, command="resume")
    assert action == Push(side="b", first=False)
    assert state["crash_streak"] == 0 and state["status"] == "running"


def test_resume_on_an_idle_chain_does_nothing():
    state, action = step(initial_state(), [RUN], None, NOW, command="resume")
    assert action is None and state == initial_state()


def test_skip_drops_the_current_run():
    state, action = step(running(), [RUN], None, NOW, command="skip")
    assert action is None and state["status"] == "idle"
    assert state["done"][-1]["name"] == "smoke" and state["done"][-1]["skipped"] is True


def test_skip_with_no_current_run_does_nothing():
    state, action = step(initial_state(), [RUN], None, NOW, command="skip")
    assert action is None and state == initial_state()


def test_api_errors_fail_once_at_the_threshold():
    state, actions = running(), []
    for _ in range(MAX_API_ERRORS + 1):
        state, action = api_error(state, "timeout")
        actions.append(action)
    assert actions[:MAX_API_ERRORS - 1] == [None] * (MAX_API_ERRORS - 1)
    assert isinstance(actions[MAX_API_ERRORS - 1], Fail)
    assert actions[MAX_API_ERRORS] is None      # one email, not one per tick
    assert state["status"] == "running"          # an outage is not a halt


def test_a_successful_step_clears_the_api_error_count():
    start = running(api_errors=3)
    state, _ = step(start, [RUN], Observation("running", None), NOW)
    assert state["api_errors"] == 0


def test_kernel_slug():
    assert kernel_slug("a0-scan-only", "b") == "deluge-a0-scan-only-b"
