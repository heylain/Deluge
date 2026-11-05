#!/usr/bin/env python
"""Write the M1 verdict tables into docs/results/m1.md (M1 design 8.4).

    python scripts/m1_verdict.py a0                   # after batch 1
    python scripts/m1_verdict.py a1                   # after batch 2
    python scripts/m1_verdict.py --dry-run rehearse-m1
    python scripts/m1_verdict.py a0 --run m1-a0-xfmr=m1-a0-xfmr-2

Reads only docs/results/runs/<name>/, which the export workflow commits --
never Kaggle. Writes only between the marker comments in m1.md; the prose
around them is written by hand. --dry-run parses one run, under any name, and
prints what it read without judging or writing anything (batch 0).

--run ROLE=RUN reads a rerun in place of the run a role names. The design
reruns a broken run under a new name (8.2, 8.3), and the tables name every
run they read, so a substitution is on the record. The rerun must be an m1-*
run: the rehearsal never stands in for one, and no run stands in for two roles.

Before judging, every run read must be what its role was pre-registered as
(deluge/eval/m1.py ROLES: mixer, budget, seed, from its final_eval.json), and
the runs a verdict judges must share one dataset. Otherwise nothing is judged
and nothing written, and each problem is named.

The early read is not binding, so an early run that never exported is left
out of its table rather than holding up the binding verdict.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deluge.eval.m1 import (A0_RUNS, A1_CFC, A1_CONV, EARLY_RUNS, ROLES,  # noqa: E402
                            a0_verdict, a1_verdict, data_problems, load_run, render_a0,
                            render_a1, render_early, repeated_runs, replace_block,
                            role_problems)

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "docs/results/runs"
PAGE = REPO / "docs/results/m1.md"


def load(root: Path, names):
    missing = [name for name in names if not (root / name / "final_eval.json").exists()]
    if missing:
        raise SystemExit(f"not exported yet: {', '.join(missing)} (under {root})")
    try:
        return [load_run(root / name) for name in names]
    except ValueError as error:
        raise SystemExit(str(error)) from None


def refuse(problems) -> None:
    if problems:
        raise SystemExit("not judging: " + "".join(f"\n  - {p}" for p in problems)
                         + "\nRerun each named run with its role's model and train "
                           "configs (design 8.2) under a new name, and read it with "
                           "--run ROLE=RUN.")


def main(argv=None, root: Path = RUNS, page: Path = PAGE) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("verdict", nargs="?", choices=("a0", "a1"))
    parser.add_argument("--dry-run", metavar="RUN")
    parser.add_argument("--run", action="append", default=[], metavar="ROLE=RUN",
                        help="read RUN in place of ROLE's run, e.g. "
                             "m1-a0-xfmr=m1-a0-xfmr-2; repeatable")
    args = parser.parse_args(argv)
    if args.dry_run:
        [run] = load(root, [args.dry_run])
        print(run)
        return 0
    if args.verdict is None:
        parser.error("name a verdict (a0, a1) or pass --dry-run RUN")

    roles = tuple(ROLES)
    reruns = dict(item.partition("=")[::2] for item in args.run)
    for role, name in reruns.items():
        if role not in roles:
            parser.error(f"--run {role}={name}: {role} is not one of {', '.join(roles)}")
        if not name.startswith("m1-"):
            parser.error(f"--run {role}={name}: a rerun must be an m1-* run")
    resolved = {role: reruns.get(role, role) for role in roles}
    repeated = repeated_runs(resolved)
    if repeated:
        parser.error("; ".join(repeated))

    def read(role_names) -> dict:
        return dict(zip(role_names, load(root, [resolved[role] for role in role_names])))

    judged = read(A0_RUNS)
    cfc_s1, cfc_s2, xfmr = (judged[role] for role in A0_RUNS)
    blocks = {}
    if args.verdict == "a0":
        refuse(role_problems(judged) + data_problems([cfc_s1, cfc_s2, xfmr]))
        a0 = a0_verdict(cfc_s1, cfc_s2, xfmr)
        blocks["a0"] = render_a0([cfc_s1, cfc_s2, xfmr], a0)
        print(f"A0: {a0.outcome}")
    else:
        exported = [role for role in EARLY_RUNS
                    if (root / resolved[role] / "final_eval.json").exists()]
        for role in sorted(set(EARLY_RUNS) - set(exported)):
            print(f"early read: {resolved[role]} is not exported; left out")
        early, binding = read(exported), read((*A1_CFC, *A1_CONV))
        refuse(role_problems({**judged, **early, **binding})
               + data_problems(list(binding.values())))
        a0 = a0_verdict(cfc_s1, cfc_s2, xfmr)
        if a0.noise0 is not None and early:     # in units of A0's noise
            blocks["early"] = render_early(cfc_s1, list(early.values()), a0.noise0)
        cfc, conv = [binding[role] for role in A1_CFC], [binding[role] for role in A1_CONV]
        a1 = a1_verdict(cfc, conv)
        blocks["a1"] = render_a1([*cfc, *conv], a1)
        print(f"A1: {a1.outcome}")
    text = page.read_text()
    for key, body in blocks.items():
        text = replace_block(text, key, body)
    page.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
