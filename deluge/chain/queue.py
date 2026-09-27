"""The chain's queue, configs/chain/runs.yaml: read, filled in, and checked.

Checked hard, because the first place a bad entry would otherwise surface is a
Kaggle session ten minutes in -- or never, if the typo is in a field only a
later session reads.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from ..config import ConfigError

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
# Kaggle titles are at most 50 characters; the kernel is "deluge-<name>-a".
MAX_NAME = 50 - len("deluge-") - len("-a")
ACCELERATORS = ("cpu", "gpu")
DATA_RE = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+:[^\s:]+$")
REQUIRED = ("name", "model", "train", "accelerator", "data")
DEFAULTS = {"model_impl": "deluge.model:build"}


def load_queue(path: Union[str, Path]) -> List[Dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    runs = raw.get("runs") or []
    if not isinstance(runs, list):
        raise ConfigError(f"{path}: `runs` must be a list")

    seen, queue = set(), []
    for index, entry in enumerate(runs):
        where = f"{path}: runs[{index}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where} must be a mapping")
        missing = [key for key in REQUIRED if key not in entry]
        if missing:
            raise ConfigError(f"{where} is missing {', '.join(missing)}")
        unknown = sorted(set(entry) - set(REQUIRED) - set(DEFAULTS))
        if unknown:
            raise ConfigError(f"{where} has unknown field(s) {', '.join(unknown)}")

        name = entry["name"]
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise ConfigError(
                f"{where}: name {name!r} must be lowercase letters, digits and "
                f"single hyphens -- it becomes the Kaggle kernel slug")
        if len(name) > MAX_NAME:
            raise ConfigError(f"{where}: name {name!r} is over {MAX_NAME} characters")
        if name in seen:
            raise ConfigError(f"{where}: name {name!r} appears more than once")
        seen.add(name)
        if entry["accelerator"] not in ACCELERATORS:
            raise ConfigError(
                f"{where}: accelerator {entry['accelerator']!r} must be one of "
                f"{', '.join(ACCELERATORS)}")
        data = entry["data"]
        if data != "synthetic" and not (isinstance(data, str) and DATA_RE.match(data)):
            raise ConfigError(
                f"{where}: data {data!r} must be `synthetic` or "
                f"`<owner>/<dataset>:<file>`")
        queue.append({**DEFAULTS, **entry})
    return queue
