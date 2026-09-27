"""The kaggle CLI adapter, against a fake runner: no network."""

import json
import subprocess
from pathlib import Path

import pytest

from deluge.chain.kaggle import KaggleCLI, KaggleError, KaggleQuotaError, git_show, render
from deluge.chain.kaggle import SESSION_PY
from deluge.chain.step import Push

# STATE's commit is fictional, so pushes read this checkout's session.py.
LOCAL = {"read_source": lambda commit: SESSION_PY.read_text()}

RUN = {"name": "smoke", "model": "m.yaml", "train": "t.yaml",
       "model_impl": "deluge.train.smoke:build", "accelerator": "gpu",
       "data": "heylain/deluge-tokens:tokens.bin",
       "commit": "abc123", "run_id": "smoke@t0"}
STATE = {"run": RUN, "session": 3, "side": "b"}


class FakeRunner:
    def __init__(self, stdout="", returncode=0, stderr="", write=None):
        self.stdout, self.returncode, self.stderr, self.write = stdout, returncode, stderr, write
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self.write and "-p" in cmd:
            self.write(Path(cmd[cmd.index("-p") + 1]))
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def metadata(tmp_path, action, state=STATE):
    render(tmp_path, "heylain", state, action)
    return json.loads((tmp_path / "kernel-metadata.json").read_text())


# ---- render ---------------------------------------------------------------- #

def test_first_push_has_no_kernel_sources_and_asks_for_a_t4(tmp_path):
    meta = metadata(tmp_path, Push("a", first=True), {**STATE, "side": "a"})
    assert meta["id"] == "heylain/deluge-smoke-a" and meta["title"] == "deluge-smoke-a"
    assert meta["kernel_sources"] == []
    assert meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["is_private"] is True and meta["enable_internet"] is True
    assert meta["dataset_sources"] == ["heylain/deluge-tokens"]


def test_handoff_sources_the_other_side(tmp_path):
    meta = metadata(tmp_path, Push("b", first=False))
    assert meta["kernel_sources"] == ["heylain/deluge-smoke-a"]


def test_cpu_synthetic_run_asks_for_nothing(tmp_path):
    state = {**STATE, "run": {**RUN, "accelerator": "cpu", "data": "synthetic"}}
    meta = metadata(tmp_path, Push("b", first=False), state)
    assert meta["enable_gpu"] is False and meta["machine_shape"] == ""
    assert meta["dataset_sources"] == []


def test_rendered_session_carries_this_push_identity(tmp_path):
    render(tmp_path, "heylain", STATE, Push("b", first=False))
    namespace = {"__name__": "rendered"}
    exec(compile((tmp_path / "session.py").read_text(), "session.py", "exec"), namespace)
    assert namespace["RUN"]["run"] == RUN
    assert namespace["RUN"]["session"] == 3 and namespace["RUN"]["side"] == "b"
    assert namespace["RUN"]["repo"] == "https://github.com/heylain/Deluge"
    # session.py refuses to train a handed-off session that finds no prior output
    assert namespace["RUN"]["first"] is False


def test_render_refuses_a_session_source_without_the_marker(tmp_path):
    with pytest.raises(KaggleError, match="RUN = {}"):
        render(tmp_path / "out", "heylain", STATE, Push("b", False),
               source="print('no marker here')\n")


# ---- status ------------------------------------------------------------------ #

@pytest.mark.parametrize("raw, expected", [
    ("KernelWorkerStatus.COMPLETE", "complete"),
    ("complete", "complete"),
    ("KernelWorkerStatus.RUNNING", "running"),
    ("queued", "queued"),
    ("ERROR", "error"),
    ("cancelAcknowledged", "cancelled"),
    ("KernelWorkerStatus.CANCEL_REQUESTED", "cancelled"),
    ("NEW_THING", "unknown"),
])
def test_status_parses_enum_and_plain_forms(raw, expected):
    # Review focus 5.
    runner = FakeRunner(stdout=f'heylain/deluge-smoke-a has status "{raw}"\n')
    assert KaggleCLI("heylain", runner=runner).status("deluge-smoke-a") == expected
    assert runner.calls[0][-1] == "heylain/deluge-smoke-a"


def test_unparseable_status_raises():
    with pytest.raises(KaggleError, match="status"):
        KaggleCLI("heylain", runner=FakeRunner(stdout="???")).status("x")


def test_a_failing_cli_raises_and_quota_is_its_own_error():
    with pytest.raises(KaggleError):
        KaggleCLI("heylain", runner=FakeRunner(returncode=1, stderr="503")).status("x")
    with pytest.raises(KaggleQuotaError):
        KaggleCLI("heylain", runner=FakeRunner(
            returncode=1, stderr="GPU quota exceeded")).status("x")


# ---- fetch ------------------------------------------------------------------- #

def test_fetch_session_json_reads_only_that_file():
    runner = FakeRunner(write=lambda d: (d / "session.json").write_text('{"exit_code": 2}'))
    assert KaggleCLI("heylain", runner=runner).fetch_session_json("s") == {"exit_code": 2}
    assert "--file-pattern" in runner.calls[0]


@pytest.mark.parametrize("write", [None, lambda d: (d / "session.json").write_text("{trunc")])
def test_missing_or_corrupt_session_json_is_none(write):
    assert KaggleCLI("heylain", runner=FakeRunner(write=write)).fetch_session_json("s") is None


# ---- push -------------------------------------------------------------------- #

def test_push_renders_and_pushes():
    seen = {}
    runner = FakeRunner(stdout="Kernel version 4 successfully pushed.",
                        write=lambda d: seen.update(files=sorted(p.name for p in d.iterdir())))
    KaggleCLI("heylain", runner=runner, **LOCAL).push(STATE, Push("b", first=False))
    assert runner.calls[0][:3] == ["kaggle", "kernels", "push"]
    assert seen["files"] == ["kernel-metadata.json", "session.py"]


def test_push_without_success_line_raises():
    # Review focus 4: the CLI has exited 0 while printing an error.
    runner = FakeRunner(stdout="Kernel push error: Notebook not found")
    with pytest.raises(KaggleError, match="Notebook not found"):
        KaggleCLI("heylain", runner=runner, **LOCAL).push(STATE, Push("b", first=False))


def test_push_refused_for_quota_is_a_quota_error():
    runner = FakeRunner(stdout="Kernel push error: You have exceeded your GPU quota")
    with pytest.raises(KaggleQuotaError):
        KaggleCLI("heylain", runner=runner, **LOCAL).push(STATE, Push("b", first=False))


def test_push_renders_the_session_script_of_the_runs_pinned_commit():
    # Review F5: a run's code is fixed for its lifetime, session.py included --
    # it calls into the pinned commit's deluge.train.
    asked, rendered = [], {}
    runner = FakeRunner(stdout="Kernel version 4 successfully pushed.",
                        write=lambda d: rendered.update(code=(d / "session.py").read_text()))
    cli = KaggleCLI("heylain", runner=runner,
                    read_source=lambda commit: asked.append(commit) or "RUN = {}\n# pinned\n")
    cli.push(STATE, Push("b", first=False))
    assert asked == ["abc123"] and "# pinned" in rendered["code"]


def test_git_show_reads_session_py_at_a_commit():
    assert "RUN_MARKER" in git_show("HEAD")
    with pytest.raises(KaggleError, match="0000000"):
        git_show("0000000000000000000000000000000000000000")
