#!/usr/bin/env python
"""Tokenize a corpus into the uint16 .bin files deluge.data reads.

  measure  Read FineWeb-edu's token_count column (GPT-2 tokens) across every
           shard -- parquet column projection, tens of MB, not the text --
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
        --revision <sha that measure printed> --tokenizer mistralai/Mistral-7B-v0.1 \\
        --prefilter <from measure> --train-tokens 2_400_000_000 --out data/long4k

Spec 7 reuses an existing 32k BPE tokenizer rather than training one. Whichever
you pick, record it: provenance is what makes a run reproducible six months
later, and retokenizing invalidates every checkpoint trained on the old ids.

measure and long read parquet directly and tokenize with the tokenizers
library: `pip install huggingface_hub pyarrow tokenizers`. stream needs
`pip install datasets transformers`.
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
from deluge.data.prep import calibrate_ratio, prepare_long, yield_estimate  # noqa: E402

MAX_TOKEN_ID = np.iinfo(TOKEN_DTYPE).max


# ---- parquet and the tokenizer (measure, long) ------------------------------- #

def source(dataset: str, files: str, revision):
    """(filesystem, the revision pinned, the matching parquet paths)."""
    from huggingface_hub import HfApi, HfFileSystem

    revision = revision or HfApi().dataset_info(dataset).sha
    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{dataset}@{revision}/{files}"))
    if not paths:
        raise SystemExit(f"no files match {files!r} in {dataset}@{revision}")
    return fs, revision, paths


def parquet_batches(fs, paths, columns, batch_size=1000):
    """(path, pyarrow RecordBatch) over every row of paths, reading only columns."""
    import pyarrow.parquet as pq

    for path in paths:
        with fs.open(path, "rb") as handle:
            for batch in pq.ParquetFile(handle).iter_batches(batch_size=batch_size,
                                                              columns=columns):
                yield path, batch


def tokenizer(name: str):
    """(vocab size, encode: list of texts -> list of id lists, no special tokens)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(name)
    vocab = tok.get_vocab_size()
    if vocab > MAX_TOKEN_ID + 1:
        raise SystemExit(f"tokenizer vocab {vocab} exceeds uint16; widen TOKEN_DTYPE "
                         f"in deluge/data/stream.py")
    return vocab, lambda texts: [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]


def measure(args) -> int:
    fs, revision, paths = source(args.dataset, args.files, args.revision)
    counts = np.concatenate([batch.column("token_count").to_numpy()
                             for _, batch in parquet_batches(fs, paths, ["token_count"], 100_000)])
    _, encode = tokenizer(args.tokenizer)
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
        "shards": len(paths), "documents": int(counts.size),
        "gpt2_tokens": int(counts.sum()), "calibration": calibration,
        "yield": yield_estimate(counts, calibration["ratio"], calibration["prefilter"]),
    }, indent=2))
    return 0


def long(args) -> int:
    fs, revision, paths = source(args.dataset, args.files, args.revision)
    vocab, encode = tokenizer(args.tokenizer)
    read, started = [], time.time()

    def batches():
        for index, (path, batch) in enumerate(parquet_batches(fs, paths, ["text", "token_count"])):
            if not read or read[-1] != path:
                read.append(path)
                print(f"[prep] {path} ({time.time() - started:.0f} s)", flush=True)
            yield batch.column("text").to_pylist(), batch.column("token_count").to_pylist()

    record = prepare_long(
        batches(), encode, args.prefilter, args.train_tokens, Path(args.out),
        {"dataset": args.dataset, "revision": revision, "files": args.files,
         "files_read": read, "tokenizer": args.tokenizer, "vocab_size": vocab})
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
                          help="dataset commit; default: the current one, which is printed")
        mode.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer id")
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
