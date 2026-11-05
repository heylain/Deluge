#!/usr/bin/env bash
# Commit what deluge/chain/export.py wrote under docs/results/runs, and push
# it. Called twice by .github/workflows/export.yml, once per step. The push
# retries because the chain's tick commits state.json to the same branch.
# -e: a failed `git add` (a path .gitignore matches) must fail the step, not
# leave the commit silently empty.
set -euo pipefail
what="$1"                       # results | checkpoints
[ -d docs/results/runs ] || exit 0
git add docs/results/runs
git diff --cached --quiet && exit 0
runs=$(git diff --cached --name-only | cut -d/ -f4 | sort -u | paste -sd, -)
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git commit --quiet -m "export: $what ($runs)"
for attempt in 1 2 3 4 5; do
  git pull --rebase --quiet && git push --quiet && exit 0
  sleep $((attempt * 5))
done
echo "::error::could not push the export after 5 attempts"
exit 1
