"""The pre-registered M1 rules (design 7.3 and 8.3), at their boundaries."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from deluge.chain.export import eval_row, write_csv
from deluge.eval.m1 import (A0_RUNS, A1_CFC, A1_CONV, EARLY_RUNS, Run, a0_verdict,
                            a1_verdict, early_read, load_run, replace_block, stability)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import m1_verdict  # noqa: E402


_UNSET = object()  # Sentinel: default best_in_run_subset to subset_loss


def run(name="r", loss=3.0, subset=None, best=_UNSET, low=(0.9,), high=(0.999,),
        steps=1000, skipped=0):
    subset_loss = loss if subset is None else subset
    best_in_run_subset = subset_loss if best is _UNSET else best
    return Run(name=name, full_loss=loss, subset_loss=subset_loss,
               best_in_run_subset=best_in_run_subset, decay_min=list(low), decay_max=list(high),
               steps=steps, skipped_steps=skipped)


# ---- stability (7.3) ------------------------------------------------------------ #

def test_a_healthy_run_is_stable():
    assert stability(run(best=3.0)) == []


@pytest.mark.parametrize("loss", [math.nan, math.inf])
def test_a_non_finite_loss_is_unstable(loss):
    assert stability(run(loss=loss))


def test_late_divergence_is_judged_at_exactly_two_hundredths():
    assert stability(run(subset=3.02, best=3.0)) == []
    assert stability(run(subset=3.0201, best=3.0))


def test_a_run_without_in_run_evals_is_unstable():
    failures = stability(run(subset=9.0, best=None))
    assert failures and any("no in-run evals" in f for f in failures)


def test_a_layer_with_every_head_forgetting_is_unstable():
    assert stability(run(low=(0.9, 0.1), high=(0.999, 0.49)))


def test_a_layer_with_every_head_frozen_is_unstable():
    assert stability(run(low=(0.99991,), high=(0.99999,)))


def test_a_layer_with_heads_on_both_sides_is_stable():
    assert stability(run(low=(0.4,), high=(0.99995,))) == []


def test_one_percent_of_skipped_steps_is_the_limit():
    assert stability(run(steps=1000, skipped=10)) == []
    assert stability(run(steps=1000, skipped=11))


# ---- A0 ------------------------------------------------------------------------- #

def test_the_noise_floor_binds_for_a_lucky_close_pair():
    verdict = a0_verdict(run("s1", 3.0), run("s2", 3.001), run("x", 3.0))
    assert verdict.noise0 == 0.01


def test_a0_passes_at_exactly_twice_noise0():
    # noise0 = 0.01 (the floor), bar 0.02: 3.0 - 2.98 rounds to 0.020000000000000018.
    assert a0_verdict(run("s1", 3.0), run("s2", 3.0), run("x", 2.98)).outcome == "pass"
    assert a0_verdict(run("s1", 3.0), run("s2", 3.0), run("x", 2.98 - 1e-6)).outcome == "behind"


def test_noise0_is_the_seed_gap_when_it_beats_the_floor():
    verdict = a0_verdict(run("s1", 3.00), run("s2", 3.05), run("x", 2.91))
    assert verdict.noise0 == pytest.approx(0.05) and verdict.outcome == "pass"


def test_stable_but_behind_is_not_a_failure():
    verdict = a0_verdict(run("s1", 3.1), run("s2", 3.1), run("x", 3.0))
    assert verdict.outcome == "behind" and verdict.gap == pytest.approx(0.1)


def test_an_unstable_a0_run_settles_the_outcome_before_any_noise():
    verdict = a0_verdict(run("s1", 3.0), run("s2", math.nan), run("x", 3.0))
    assert verdict.outcome == "unstable" and verdict.noise0 is None
    assert list(verdict.unstable) == ["s2"]


def test_the_early_read_is_in_units_of_noise0():
    assert early_read(run(loss=3.0), run(loss=3.03), 0.01) == pytest.approx(3.0)


# ---- A1 ------------------------------------------------------------------------- #

def arms(cfc, conv):
    return ([run(f"cfc{i}", v) for i, v in enumerate(cfc)],
            [run(f"conv{i}", v) for i, v in enumerate(conv)])


def test_go_at_exactly_two_estimated_standard_errors():
    bar = 2 * 0.01 / math.sqrt(3)                       # floor-bound noise
    assert a1_verdict(*arms([3.0] * 3, [3.0 + bar] * 3)).outcome == "GO"
    assert a1_verdict(*arms([3.0] * 3, [3.0 + bar - 1e-6] * 3)).outcome == "NO-GO"


def test_the_noise_uses_the_n_minus_one_sample_sd():
    verdict = a1_verdict(*arms([3.00, 3.02, 3.04], [3.1] * 3))
    # stdev([3.00, 3.02, 3.04]) = 0.02 (n - 1); pstdev would give 0.0163.
    assert verdict.noise == pytest.approx(math.sqrt(2) * 0.02)
    assert verdict.gap == pytest.approx(0.08)
    assert verdict.bar == pytest.approx(2 * verdict.noise / math.sqrt(3))


def test_the_a1_floor_binds_when_the_seeds_agree():
    assert a1_verdict(*arms([3.0, 3.001, 3.002], [3.1] * 3)).noise == 0.01


def test_an_unstable_binding_run_is_no_verdict_not_a_no_go():
    cfc, conv = arms([3.0] * 3, [3.5] * 3)
    conv[2] = run("conv2", 3.5, skipped=500)
    verdict = a1_verdict(cfc, conv)
    assert verdict.outcome == "no verdict" and verdict.noise is None


def test_the_binding_verdict_needs_three_seeds():
    with pytest.raises(ValueError, match="three seeds"):
        a1_verdict(*arms([3.0] * 2, [3.1] * 2))


# ---- reading exports and writing m1.md --------------------------------------------- #

def export(root, name, loss, in_run=(3.5, 3.4)):
    """Write a run the way deluge/chain/export.py does."""
    out = root / name
    out.mkdir(parents=True)
    (out / "final_eval.json").write_text(json.dumps({
        "step": 1907, "full_loss": loss, "subset_loss": loss, "skipped_steps": 0,
        "buckets": {"0-512": loss + 0.2, "512-1536": loss, "1536-4096": loss - 0.1},
        "decay": {"min": [0.9], "median": [0.95], "max": [0.999]}, "state_rms": [1.0]}))
    write_csv(out / "evals.csv", [eval_row({"step": 190 * (i + 1), "subset_loss": v})
                                  for i, v in enumerate(in_run)])


def test_a_run_is_read_back_from_what_the_exporter_writes(tmp_path):
    export(tmp_path, "m1-a0-cfc-s1", 3.3)
    loaded = load_run(tmp_path / "m1-a0-cfc-s1")
    assert loaded.full_loss == 3.3 and loaded.best_in_run_subset == 3.4
    assert loaded.decay_max == [0.999] and loaded.steps == 1907
    assert loaded.buckets["1536-4096"] == pytest.approx(3.2)


def test_replace_block_writes_only_between_the_markers():
    text = "intro\n<!-- m1_verdict:a0 -->\nold\n<!-- /m1_verdict:a0 -->\noutro\n"
    assert replace_block(text, "a0", "new") == (
        "intro\n<!-- m1_verdict:a0 -->\nnew\n<!-- /m1_verdict:a0 -->\noutro\n")
    with pytest.raises(ValueError, match="a1"):
        replace_block(text, "a1", "new")


def test_the_results_page_has_every_marker():
    text = (REPO / "docs/results/m1.md").read_text()
    for key in ("a0", "early", "a1"):
        assert f"<!-- m1_verdict:{key} -->" in text and f"<!-- /m1_verdict:{key} -->" in text


def test_the_script_writes_the_a0_and_a1_tables(tmp_path):
    root = tmp_path / "runs"
    for name, loss in zip(A0_RUNS, (3.30, 3.31, 3.29)):
        export(root, name, loss)
    for name in EARLY_RUNS:
        export(root, name, 3.35)
    for name, loss in [*zip(A1_CFC, (3.00, 3.01, 3.02)), *zip(A1_CONV, (3.05, 3.06, 3.07))]:
        export(root, name, loss)
    page = tmp_path / "m1.md"
    page.write_text((REPO / "docs/results/m1.md").read_text())

    assert m1_verdict.main(["a0"], root=root, page=page) == 0
    assert "**A0: pass.**" in page.read_text()
    assert m1_verdict.main(["a1"], root=root, page=page) == 0
    text = page.read_text()
    assert "**A1: GO.**" in text and "`m1-a1s-lru` - CfC: +5.00 noise0" in text
    assert "## Caveats" in text                     # the prose is untouched


def test_the_dry_run_reads_any_name_and_writes_nothing(tmp_path, capsys):
    export(tmp_path, "rehearse-m1", 3.3)
    page = tmp_path / "m1.md"
    page.write_text("untouched")
    assert m1_verdict.main(["--dry-run", "rehearse-m1"], root=tmp_path, page=page) == 0
    out = capsys.readouterr().out
    assert "rehearse-m1" in out and "stability" not in out and page.read_text() == "untouched"


def test_a_verdict_with_runs_missing_names_them(tmp_path):
    with pytest.raises(SystemExit, match="m1-a0-cfc-s1"):
        m1_verdict.main(["a0"], root=tmp_path, page=tmp_path / "m1.md")


def test_a_rerun_under_a_new_name_stands_in_for_its_role(tmp_path):
    # Design 8.2-8.3 rerun a broken run under a new name; the verdict must be
    # able to read it, and the table must say which run it read.
    root = tmp_path / "runs"
    for name, loss in zip(A0_RUNS, (3.30, 3.31, 3.29)):
        export(root, name, loss)
    export(root, "m1-a0-xfmr-2", 3.20)
    page = tmp_path / "m1.md"
    page.write_text((REPO / "docs/results/m1.md").read_text())
    assert m1_verdict.main(["a0", "--run", "m1-a0-xfmr=m1-a0-xfmr-2"],
                           root=root, page=page) == 0
    text = page.read_text()
    assert "| `m1-a0-xfmr-2` |" in text and "CfC s1 - transformer = +0.1000" in text
    export(root, "rehearse-m1", 3.0)
    for bad in ("m1-a0-xfrm=m1-a0-xfmr-2", "m1-a0-xfmr=rehearse-m1"):
        with pytest.raises(SystemExit):
            m1_verdict.main(["a0", "--run", bad], root=root, page=page)


def test_the_binding_verdict_does_not_wait_on_the_early_read(tmp_path):
    # The early runs bind nothing. One skipped after crashing never exports,
    # and it must not hold up the verdict the project turns on.
    root = tmp_path / "runs"
    for name, loss in [*zip(A0_RUNS, (3.30, 3.31, 3.29)), (EARLY_RUNS[0], 3.35),
                       *zip(A1_CFC, (3.00, 3.01, 3.02)), *zip(A1_CONV, (3.05, 3.06, 3.07))]:
        export(root, name, loss)
    page = tmp_path / "m1.md"
    page.write_text((REPO / "docs/results/m1.md").read_text())
    assert m1_verdict.main(["a1"], root=root, page=page) == 0
    text = page.read_text()
    assert "**A1: GO.**" in text and f"`{EARLY_RUNS[0]}` - CfC" in text
    assert f"`{EARLY_RUNS[1]}` - CfC" not in text
