#!/usr/bin/env python
"""Render the Kaggle CPU kernel that builds the M1 dataset (M1 design 6.1 step 6).

The kernel, deluge-prep-long4k, is private, CPU-only and has internet. It
clones this repo at a pinned commit and runs `scripts/prepare_data.py long`
into /kaggle/working, so the hours of download and tokenizing run on Kaggle
rather than on the home PC. Its output becomes the private dataset every M1 arm
attaches (docs/kaggle.md, "Data").

    python scripts/kaggle_prep.py --owner heylaine --commit <sha on GitHub> \\
        --dir /tmp/prep-kernel -- \\
        --dataset HuggingFaceFW/fineweb-edu --files 'sample/10BT/*.parquet' \\
        --revision <sha> --tokenizer mistralai/Mistral-7B-v0.1 \\
        --tokenizer-revision <sha> --prefilter <n> --train-tokens 2_400_000_000
    KAGGLE_CONFIG_DIR=~/Downloads kaggle kernels push -p /tmp/prep-kernel

Everything after `--` is passed to `prepare_data.py long`; --out and
--builder-commit (the --commit it checks out) are the kernel's.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = "https://github.com/heylain/Deluge"
SLUG = "deluge-prep-long4k"
# The versions in .venv, which `prepare_data.py measure` calibrates the
# pre-filter with. Unpinned, pip keeps whatever Kaggle preinstalled, and
# another tokenizers can tokenize the same text differently.
PIP_PINS = ("huggingface_hub==1.33.0", "pyarrow==25.0.1", "tokenizers==0.23.2")

KERNEL = '''"""Builds the Deluge M1 dataset. Rendered by scripts/kaggle_prep.py."""
import json
import subprocess
import sys

PARAMS = json.loads({params!r})
SRC = "/tmp/deluge-src"
subprocess.run(["git", "clone", "--quiet", PARAMS["repo"], SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", PARAMS["commit"]], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", SRC,
                *PARAMS["pip"]], check=True)
subprocess.run([sys.executable, SRC + "/scripts/prepare_data.py", "long",
                *PARAMS["args"], "--builder-commit", PARAMS["commit"],
                "--out", "/kaggle/working"], check=True)
'''


def render(dest: Path, owner: str, commit: str, prep_args: list) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"--commit must be a 40-hex sha, as git rev-parse HEAD prints: "
                         f"the kernel records it as builder_commit, not {commit!r}")
    if "--out" in prep_args:
        raise ValueError("--out is the kernel's: /kaggle/working, whose files become the dataset")
    if "--builder-commit" in prep_args:
        raise ValueError("--builder-commit is the kernel's: the --commit it checks out")
    dest.mkdir(parents=True, exist_ok=True)
    params = json.dumps({"repo": REPO, "commit": commit, "pip": PIP_PINS, "args": prep_args})
    (dest / "prep_kernel.py").write_text(KERNEL.format(params=params))
    (dest / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{SLUG}", "title": SLUG, "code_file": "prep_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_internet": True, "machine_shape": "",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--owner", required=True, help="Kaggle username")
    parser.add_argument("--commit", required=True,
                        help="the full sha of a commit the kernel can clone from GitHub")
    parser.add_argument("--dir", required=True, type=Path, help="where to render the kernel")
    parser.add_argument("prep_args", nargs=argparse.REMAINDER,
                        help="-- then the arguments for prepare_data.py long")
    args = parser.parse_args(argv)
    prep_args = args.prep_args[1:] if args.prep_args[:1] == ["--"] else args.prep_args
    render(args.dir, args.owner, args.commit, prep_args)
    print(f"rendered {args.dir}; push it with: kaggle kernels push -p {args.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
