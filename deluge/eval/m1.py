"""The M1 verdicts: A0's gate and A1's go/no-go (M1 design 7.3 and 8.3).

Pre-registered: these rules were fixed before any result existed, and this
module is them and nothing more. Stability is judged first, with fixed
thresholds only; seed noise is then computed from stable runs only, so a
broken run can neither inflate the bar nor pass its own check.

Reads what deluge/chain/export.py committed to docs/results/runs/<name>/:
final_eval.json and evals.csv. scripts/m1_verdict.py is the command line.

Comparisons carry EPS, so a value exactly on a boundary lands on the side the
rule names (<= for A0, >= for A1) whatever the float rounding.
"""

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

EPS = 1e-12
NOISE_FLOOR = 0.01            # nats; stops a lucky close pair setting the bar
LATE_DIVERGENCE = 0.02        # nats; final subset <= best in-run subset + this
AMNESIA = 0.5                 # a layer with every head below: amnesia
FROZEN = 0.9999               # every head above: ~10k-token memory, b ~ 0.014
MAX_SKIPPED = 0.01            # of steps, skipped by the fp16 scaler

A0_RUNS = ("m1-a0-cfc-s1", "m1-a0-cfc-s2", "m1-a0-xfmr")
EARLY_RUNS = ("m1-a1s-conv", "m1-a1s-lru")
A1_CFC = ("m1-a1d-cfc-s1", "m1-a1d-cfc-s2", "m1-a1d-cfc-s3")
A1_CONV = ("m1-a1d-conv-s1", "m1-a1d-conv-s2", "m1-a1d-conv-s3")


@dataclass(frozen=True)
class Run:
    name: str
    full_loss: float
    subset_loss: float
    best_in_run_subset: Optional[float]      # None: no in-run evals
    decay_min: List[float]                   # per recurrent layer, over heads
    decay_max: List[float]
    steps: int
    skipped_steps: int
    buckets: Dict[str, Optional[float]] = field(default_factory=dict)


def load_run(directory: Path) -> Run:
    final = json.loads((directory / "final_eval.json").read_text())
    in_run: List[float] = []
    evals = directory / "evals.csv"
    if evals.exists():
        with open(evals, newline="") as handle:
            in_run = [float(row["subset_loss"]) for row in csv.DictReader(handle)]
    decay = final.get("decay") or {}
    return Run(name=directory.name, full_loss=float(final["full_loss"]),
               subset_loss=float(final["subset_loss"]),
               best_in_run_subset=min(in_run) if in_run else None,
               decay_min=list(decay.get("min", [])), decay_max=list(decay.get("max", [])),
               steps=int(final["step"]), skipped_steps=int(final["skipped_steps"]),
               buckets=dict(final.get("buckets") or {}))


def stability(run: Run) -> List[str]:
    """The criteria of design 7.3 a run fails; empty means stable."""
    failures = []
    if not math.isfinite(run.full_loss):
        failures.append(f"final loss is {run.full_loss}")
    if run.best_in_run_subset is not None and not (
            run.subset_loss <= run.best_in_run_subset + LATE_DIVERGENCE + EPS):
        failures.append(f"late divergence: final subset loss {run.subset_loss:.4f} > "
                        f"best in-run {run.best_in_run_subset:.4f} + {LATE_DIVERGENCE}")
    for layer, (low, high) in enumerate(zip(run.decay_min, run.decay_max)):
        if high < AMNESIA:
            failures.append(f"recurrent layer {layer}: every head below a = {AMNESIA}")
        if low > FROZEN:
            failures.append(f"recurrent layer {layer}: every head above a = {FROZEN}")
    if run.skipped_steps > MAX_SKIPPED * run.steps + EPS:
        failures.append(f"{run.skipped_steps} of {run.steps} steps skipped by the "
                        f"loss scaler, over {MAX_SKIPPED:.0%}")
    return failures


def _unstable(runs: Sequence[Run]) -> Dict[str, List[str]]:
    return {run.name: failures for run in runs if (failures := stability(run))}


@dataclass(frozen=True)
class A0Verdict:
    outcome: str                       # "pass" | "behind" | "unstable"
    noise0: Optional[float] = None
    gap: Optional[float] = None        # CfC s1 - transformer; > 0 means CfC is behind
    unstable: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def bar(self) -> Optional[float]:
        return None if self.noise0 is None else 2 * self.noise0


def a0_verdict(cfc_s1: Run, cfc_s2: Run, transformer: Run) -> A0Verdict:
    """Pass iff all three are stable and CfC s1 - transformer <= 2 noise0.

    "behind" -- stable, but past the bar -- is not a fail: it is diagnosed,
    then either fixed and rerun or carried into A1 as a caveat (design 8.3).
    """
    unstable = _unstable((cfc_s1, cfc_s2, transformer))
    if unstable:
        return A0Verdict("unstable", unstable=unstable)
    noise0 = max(abs(cfc_s1.full_loss - cfc_s2.full_loss), NOISE_FLOOR)
    gap = cfc_s1.full_loss - transformer.full_loss
    return A0Verdict("pass" if gap <= 2 * noise0 + EPS else "behind", noise0, gap)


def early_read(cfc_s1: Run, other: Run, noise0: float) -> float:
    """other - CfC at seed 1234, in units of noise0. Reported; never acted on."""
    return (other.full_loss - cfc_s1.full_loss) / noise0


@dataclass(frozen=True)
class A1Verdict:
    outcome: str                       # "GO" | "NO-GO" | "no verdict"
    noise: Optional[float] = None
    gap: Optional[float] = None        # mean(conv) - mean(CfC); > 0 means CfC is better
    bar: Optional[float] = None
    unstable: Dict[str, List[str]] = field(default_factory=dict)


def a1_verdict(cfc: Sequence[Run], conv: Sequence[Run]) -> A1Verdict:
    """GO iff all six are stable and mean(conv) - mean(CfC) >= 2 noise / sqrt(3),
    noise = max(sqrt(2) s, NOISE_FLOOR) with s the n - 1 sample SD of the CfC
    losses. An unstable run means no verdict -- not a no-go."""
    if len(cfc) != 3 or len(conv) != 3:
        raise ValueError("the binding verdict needs three seeds per arm")
    unstable = _unstable((*cfc, *conv))
    if unstable:
        return A1Verdict("no verdict", unstable=unstable)
    s = statistics.stdev(run.full_loss for run in cfc)
    noise = max(math.sqrt(2) * s, NOISE_FLOOR)
    gap = (statistics.fmean(run.full_loss for run in conv)
           - statistics.fmean(run.full_loss for run in cfc))
    bar = 2 * noise / math.sqrt(3)
    return A1Verdict("GO" if gap >= bar - EPS else "NO-GO", noise, gap, bar)


# ---- docs/results/m1.md --------------------------------------------------------- #

def runs_table(runs: Sequence[Run]) -> str:
    buckets = sorted({b for run in runs for b in run.buckets},
                     key=lambda b: int(b.split("-")[0]))
    head = ["run", "full loss", "subset", "best in-run", *buckets, "skipped", "stable"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for run in runs:
        failures = stability(run)
        cells = [f"`{run.name}`", f"{run.full_loss:.4f}", f"{run.subset_loss:.4f}",
                 "-" if run.best_in_run_subset is None else f"{run.best_in_run_subset:.4f}",
                 *("-" if run.buckets.get(b) is None else f"{run.buckets[b]:.4f}"
                   for b in buckets),
                 f"{run.skipped_steps}/{run.steps}",
                 "yes" if not failures else "no: " + "; ".join(failures)]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render_a0(runs: Sequence[Run], verdict: A0Verdict) -> str:
    lines = [runs_table(runs), ""]
    if verdict.outcome == "unstable":
        lines.append("**A0: unstable.** Debug, fix, and rerun under new names.")
    else:
        lines += [f"noise0 = {verdict.noise0:.4f} nats; CfC s1 - transformer = "
                  f"{verdict.gap:+.4f}; bar 2 x noise0 = {verdict.bar:.4f}.", "",
                  "**A0: pass.**" if verdict.outcome == "pass" else
                  "**A0: stable but behind.** Diagnose before rerunning anything "
                  "(design 8.3)."]
    return "\n".join(lines)


def render_early(cfc_s1: Run, others: Sequence[Run], noise0: float) -> str:
    lines = [runs_table([cfc_s1, *others]), ""]
    lines += [f"- `{run.name}` - CfC: {early_read(cfc_s1, run, noise0):+.2f} noise0"
              for run in others]
    lines += ["", "Not binding: an early read triggers nothing."]
    return "\n".join(lines)


def render_a1(runs: Sequence[Run], verdict: A1Verdict) -> str:
    lines = [runs_table(runs), ""]
    if verdict.outcome == "no verdict":
        lines.append("**No verdict:** a binding run is unstable. Debug, fix, and "
                     "rerun under new names; instability is not a no-go.")
    else:
        lines += [f"noise = {verdict.noise:.4f} nats; gap = mean(conv) - mean(CfC) = "
                  f"{verdict.gap:+.4f}; bar 2 x noise / sqrt(3) = {verdict.bar:.4f}.", "",
                  f"**A1: {verdict.outcome}.**"]
    return "\n".join(lines)


def replace_block(text: str, key: str, body: str) -> str:
    """Put body between <!-- m1_verdict:key --> and <!-- /m1_verdict:key -->,
    leaving everything outside the markers alone."""
    pattern = re.compile(rf"(<!-- m1_verdict:{key} -->\n).*?(\n<!-- /m1_verdict:{key} -->)",
                         re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"no <!-- m1_verdict:{key} --> block to write into")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text, count=1)
