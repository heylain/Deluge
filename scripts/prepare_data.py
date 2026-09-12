#!/usr/bin/env python
"""Tokenize a corpus into the flat uint16 .bin that deluge.data reads.

Streams by default: the dataset is never materialised on disk, only the token
file is. That matters on Kaggle, where the working directory is the budget, and
on a laptop with 27 GB free.

    python scripts/prepare_data.py \\
        --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \\
        --tokenizer mistralai/Mistral-7B-v0.1 \\
        --tokens 600_000_000 --out data/tokens.bin

Spec 7 reuses an existing 32k BPE tokenizer rather than training one. Whichever
you pick, record it: the .json written alongside the .bin is what makes a run
reproducible six months later, and retokenizing invalidates every checkpoint
trained on the old ids.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Run from anywhere: python puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deluge.data import TOKEN_DTYPE  # noqa: E402

MAX_TOKEN_ID = np.iinfo(TOKEN_DTYPE).max


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset id")
    parser.add_argument("--name", default=None, help="dataset config name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer id")
    parser.add_argument("--tokens", type=int, required=True,
                        help="stop after this many tokens")
    parser.add_argument("--out", required=True, help="output .bin path")
    parser.add_argument("--batch", type=int, default=1000,
                        help="documents per tokenizer call")
    parser.add_argument("--no-streaming", action="store_true",
                        help="download the dataset instead of streaming it")
    args = parser.parse_args(argv)

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        print("needs `pip install datasets transformers`", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.vocab_size > MAX_TOKEN_ID:
        print(f"tokenizer vocab {tokenizer.vocab_size} exceeds uint16; widen "
              f"TOKEN_DTYPE in deluge/data.py", file=sys.stderr)
        return 1
    # Documents are concatenated into one stream, so they need a boundary token
    # or the model learns to run one document into the next.
    separator = tokenizer.eos_token_id
    if separator is None:
        print("tokenizer has no eos_token_id to separate documents", file=sys.stderr)
        return 1

    dataset = load_dataset(args.dataset, args.name, split=args.split,
                           streaming=not args.no_streaming)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    started = time.time()

    with open(out, "wb") as handle:
        batch: list[str] = []

        def flush(batch: list[str]) -> int:
            encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
            flat = [t for ids in encoded for t in (*ids, separator)]
            remaining = args.tokens - written
            chunk = np.asarray(flat[:remaining], dtype=TOKEN_DTYPE)
            chunk.tofile(handle)
            return len(chunk)

        for record in dataset:
            batch.append(record[args.text_field])
            if len(batch) < args.batch:
                continue
            written += flush(batch)
            batch = []
            elapsed = time.time() - started
            print(f"\r{written/1e6:.1f}M / {args.tokens/1e6:.0f}M tokens "
                  f"({written/elapsed/1e3:.0f}k tok/s)", end="", flush=True)
            if written >= args.tokens:
                break
        if batch and written < args.tokens:
            written += flush(batch)

    print()
    meta = {
        "dataset": args.dataset, "name": args.name, "split": args.split,
        "tokenizer": args.tokenizer, "vocab_size": tokenizer.vocab_size,
        "eos_token_id": separator, "tokens": written,
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "bytes": out.stat().st_size,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {written/1e6:.1f}M tokens to {out} "
          f"({out.stat().st_size/1e9:.2f} GB) and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
