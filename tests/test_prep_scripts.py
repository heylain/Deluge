"""The data-prep command lines: argument wiring only, no network."""

import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import kaggle_prep  # noqa: E402
import prepare_data  # noqa: E402


def test_prepare_data_has_the_three_modes(capsys):
    with pytest.raises(SystemExit) as exit_:
        prepare_data.main(["--help"])
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    assert all(mode in out for mode in ("measure", "long", "stream"))


def test_the_long_mode_needs_a_prefilter_and_a_target():
    with pytest.raises(SystemExit):
        prepare_data.main(["long", "--dataset", "d", "--files", "f", "--tokenizer", "t",
                           "--out", "o"])


def test_the_prep_kernel_is_a_private_cpu_kernel_with_internet(tmp_path):
    kaggle_prep.main(["--owner", "heylaine", "--commit", "abc123", "--dir", str(tmp_path),
                      "--", "--dataset", "HuggingFaceFW/fineweb-edu", "--prefilter", "2700"])
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "heylaine/deluge-prep-long4k" and meta["is_private"] is True
    assert meta["enable_gpu"] is False and meta["enable_internet"] is True
    code = (tmp_path / "prep_kernel.py").read_text()
    ast.parse(code)
    params = json.loads(ast.literal_eval(code.split("json.loads(")[1].split(")\n")[0]))
    assert params["commit"] == "abc123"
    assert params["args"] == ["--dataset", "HuggingFaceFW/fineweb-edu", "--prefilter", "2700"]
    assert '"--out", "/kaggle/working"' in code


def test_the_prep_kernel_owns_the_output_directory(tmp_path):
    with pytest.raises(ValueError, match="--out"):
        kaggle_prep.render(tmp_path, "o", "c", ["--out", "/elsewhere"])
