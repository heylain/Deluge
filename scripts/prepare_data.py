#!/usr/bin/env python
"""Tokenize a corpus into the uint16 .bin files deluge.data reads.

  measure  Read FineWeb-edu's token_count column (GPT-2 tokens) across every
           shard -- parquet column projection, not the text: on the 10BT
           sample about 50 MiB, in about 10,000 small range requests --
           calibrate the GPT-2 -> training-tokenizer ratio on a sample of
           documents, and report the yield of whole 4,097-token pieces and a
           loose pre-filter. Run it before fixing any dataset size (M1 design
           6.1 step 1).
  long     The M1 dataset: documents of at least 4,097 tokens, cut into whole
           pieces, split 99/1 by a hash of their text, written as train.bin,
           heldout.bin and provenance.json (deluge/data/prep.py says why).
  stream   Documents concatenated into one stream with an EOS separator, for
           short-context work. The original mode of this script.

    python scripts/prepare_data.py measure \\
        --dataset HuggingFaceFW/fineweb-edu --files 'sample/10BT/*.parquet' \\
        --tokenizer mistralai/Mistral-7B-v0.1

    python scripts/prepare_data.py long \\
        --dataset HuggingFaceFW/fineweb-edu --files 'sample/10BT/*.parquet' \\
        --revision <the revision measure printed> --tokenizer mistralai/Mistral-7B-v0.1 \\
        --tokenizer-revision <the tokenizer_revision measure printed> \\
        --prefilter <from measure> --train-tokens 2_400_000_000 --out data/long4k

Spec 7 reuses an existing 32k BPE tokenizer rather than training one. Whichever
you pick, record it: provenance is what makes a run reproducible six months
later, and retokenizing invalidates every checkpoint trained on the old ids.
So long records what the ids depend on: the dataset and tokenizer at commit
shas (a branch name is resolved to one), the tokenizers version, the commit of
the code that built the data, and the target. A long build that runs out of
documents before --train-tokens exits non-zero and moves nothing into place.

measure and long read parquet directly and tokenize with the tokenizers
library. Install the versions the Kaggle kernel pins (PIP_PINS in
scripts/kaggle_prep.py), so measure calibrates with what long builds with.
stream needs `pip install datasets transformers`.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Run from anywhere: python puts scripts/ on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deluge.data import TOKEN_DTYPE  # noqa: E402
from deluge.data.prep import (ShortBuild, calibrate_ratio, prepare_long,  # noqa: E402
                              yield_estimate)

MAX_TOKEN_ID = np.iinfo(TOKEN_DTYPE).max
# measure's token_count pass is one small range request per row group, bound by
# round trips rather than bandwidth, so shards are read concurrently. 16 reads
# the 10BT sample's 14 shards all at once.
MEASURE_THREADS = 16


# ---- parquet and the tokenizer (measure, long) ------------------------------- #

def source(dataset: str, files: str, revision):
    """(filesystem, the revision pinned, the matching parquet paths).

    revision -- a sha, a branch, or None for the current one -- is always
    resolved to a commit sha: "main" in provenance pins nothing."""
    from huggingface_hub import HfApi, HfFileSystem

    revision = HfApi().dataset_info(dataset, revision=revision).sha
    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{dataset}@{revision}/{files}"))
    if not paths:
        raise SystemExit(f"no files match {files!r} in {dataset}@{revision}")
    return fs, revision, paths


def parquet_batches(fs, paths, columns, batch_size=1000, **open_options):
    """(path, pyarrow RecordBatch) over every row of paths, reading only columns.
    open_options go to fs.open."""
    import pyarrow.parquet as pq

    for path in paths:
        with fs.open(path, "rb", **open_options) as handle:
            for batch in pq.ParquetFile(handle).iter_batches(batch_size=batch_size,
                                                              columns=columns):
                yield path, batch


def token_counts(fs, paths) -> np.ndarray:
    """Every row's token_count, in path order.

    Opened with no cache: fsspec's default read-ahead fetches 5 MiB per read,
    and FineWeb-edu's row groups sit ~2.8 MB apart, so each ~4 KB column chunk
    would pull its row group's text with it: the whole sample (~28.5 GB for
    10BT) for ~41 MiB of counts."""
    def shard(path):
        return [batch.column("token_count").to_numpy()
                for _, batch in parquet_batches(fs, [path], ["token_count"], 100_000,
                                                cache_type="none")]

    with ThreadPoolExecutor(max_workers=min(MEASURE_THREADS, len(paths))) as pool:
        return np.concatenate([counts for shard_counts in pool.map(shard, paths)
                               for counts in shard_counts])


def tokenizer(name: str, revision):
    """(its provenance, encode: list of texts -> list of id lists, no special tokens).

    Loaded at a commit sha, resolved from revision (None: the current one), and
    recorded with the tokenizers version: a change to either can change the ids."""
    import tokenizers
    from huggingface_hub import HfApi

    revision = HfApi().model_info(name, revision=revision).sha
    tok = tokenizers.Tokenizer.from_pretrained(name, revision=revision)
    vocab = tok.get_vocab_size()
    if vocab > MAX_TOKEN_ID + 1:
        raise SystemExit(f"tokenizer vocab {vocab} exceeds uint16; widen TOKEN_DTYPE "
                         f"in deluge/data/stream.py")
    record = {"tokenizer": name, "tokenizer_revision": revision,
              "tokenizers_version": tokenizers.__version__, "vocab_size": vocab}
    return record, lambda texts: [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]


def head_commit(repo: Path = ROOT):
    """HEAD of the git checkout at repo, or None if it is not one."""
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def measure(args) -> int:
    fs, revision, paths = source(args.dataset, args.files, args.revision)
    counts = token_counts(fs, paths)
    tokenized, encode = tokenizer(args.tokenizer, args.tokenizer_revision)
    texts, gpt2 = [], []
    for _, batch in parquet_batches(fs, paths[:1], ["text", "token_count"]):
        for text, n in zip(batch.column("text").to_pylist(),
                           batch.column("token_count").to_pylist()):
            if n >= args.calibrate_min:
                texts.append(text)
                gpt2.append(n)
        if len(texts) >= args.calibrate:
            break
    calibration = calibrate_ratio(gpt2, [len(ids) for ids in encode(texts)])
    print(json.dumps({
        "dataset": args.dataset, "revision": revision, "files": args.files,
        **tokenized, "shards": len(paths), "documents": int(counts.size),
        "gpt2_tokens": int(counts.sum()), "calibration": calibration,
        "yield": yield_estimate(counts, calibration["ratio"], calibration["prefilter"]),
    }, indent=2))
    return 0


def long(args) -> int:
    fs, revision, paths = source(args.dataset, args.files, args.revision)
    tokenized, encode = tokenizer(args.tokenizer, args.tokenizer_revision)
    read, started = [], time.time()

    def batches():
        for index, (path, batch) in enumerate(parquet_batches(fs, paths, ["text", "token_count"])):
            if not read or read[-1] != path:
                read.append(path)
                print(f"[prep] {path} ({time.time() - started:.0f} s)", flush=True)
            yield batch.column("text").to_pylist(), batch.column("token_count").to_pylist()

    try:
        record = prepare_long(
            batches(), encode, args.prefilter, args.train_tokens, Path(args.out),
            {"dataset": args.dataset, "revision": revision, "files": args.files,
             "files_read": read, **tokenized,
             "builder_commit": args.builder_commit or head_commit()})
    except ShortBuild as short:
        raise SystemExit(f"[prep] {short}. Build from a larger --files sample "
                         f"than {args.files!r} (for FineWeb-edu, "
                         f"'sample/100BT/*.parquet').") from None
    print(json.dumps(record, indent=2))
    return 0


# ---- stream (the original mode) ------------------------------------------------ #

def stream(args) -> int:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        print("needs `pip install datasets transformers`", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.vocab_size > MAX_TOKEN_ID:
        print(f"tokenizer vocab {tok.vocab_size} exceeds uint16; widen "
              f"TOKEN_DTYPE in deluge/data/stream.py", file=sys.stderr)
        return 1
    # Documents are concatenated into one stream, so they need a boundary token
    # or the model learns to run one document into the next.
    separator = tok.eos_token_id
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
            encoded = tok(batch, add_special_tokens=False)["input_ids"]
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
        "tokenizer": args.tokenizer, "vocab_size": tok.vocab_size,
        "eos_token_id": separator, "tokens": written,
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "bytes": out.stat().st_size,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {written/1e6:.1f}M tokens to {out} "
          f"({out.stat().st_size/1e9:.2f} GB) and {out.with_suffix('.json')}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    modes = parser.add_subparsers(dest="mode", required=True)

    for name in ("measure", "long"):
        mode = modes.add_parser(name)
        mode.add_argument("--dataset", required=True, help="HuggingFace dataset id")
        mode.add_argument("--files", required=True,
                          help="parquet glob inside the dataset repo, e.g. 'sample/10BT/*.parquet'")
        mode.add_argument("--revision", default=None,
                          help="dataset commit or branch, resolved to a sha; default: "
                               "the current one, which is printed")
        mode.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer id")
        mode.add_argument("--tokenizer-revision", default=None,
                          help="tokenizer commit or branch, resolved to a sha; default: "
                               "the current one, which is printed")
    measure_mode, long_mode = modes.choices["measure"], modes.choices["long"]
    measure_mode.add_argument("--calibrate", type=int, default=3000,
                              help="documents to tokenize for the ratio")
    measure_mode.add_argument("--calibrate-min", type=int, default=2000,
                              help="GPT-2 length a calibration document needs: "
                                   "the ratio that matters is long documents'")
    long_mode.add_argument("--prefilter", type=int, required=True,
                           help="GPT-2 token_count below which a document is skipped "
                                "untokenized (measure suggests one)")
    long_mode.add_argument("--train-tokens", type=int, required=True,
                           help="stop once train.bin holds this many")
    long_mode.add_argument("--out", required=True, help="output directory")
    long_mode.add_argument("--builder-commit", default=None,
                           help="commit of the code building the data, for provenance; "
                                "default: HEAD of this script's checkout (null outside one)")

    stream_mode = modes.add_parser("stream")
    stream_mode.add_argument("--dataset", required=True, help="HuggingFace dataset id")
    stream_mode.add_argument("--name", default=None, help="dataset config name")
    stream_mode.add_argument("--split", default="train")
    stream_mode.add_argument("--text-field", default="text")
    stream_mode.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer id")
    stream_mode.add_argument("--tokens", type=int, required=True,
                             help="stop after this many tokens")
    stream_mode.add_argument("--out", required=True, help="output .bin path")
    stream_mode.add_argument("--batch", type=int, default=1000,
                             help="documents per tokenizer call")
    stream_mode.add_argument("--no-streaming", action="store_true",
                             help="download the dataset instead of streaming it")

    args = parser.parse_args(argv)
    return {"measure": measure, "long": long, "stream": stream}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
