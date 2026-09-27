"""Unattended training across Kaggle's 12 h sessions.

queue.py reads configs/chain/runs.yaml; step.py decides what to do next and
does no I/O; kaggle.py talks to the kaggle CLI; orchestrate.py is one tick of
.github/workflows/chain.yml; session.py is what a Kaggle kernel runs. Design:
docs/superpowers/specs/2026-09-27-kaggle-chain-design.md.

Importable without torch: the orchestrator runs on a GitHub runner with the
base install only.
"""
