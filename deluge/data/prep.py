"""Long-document data preparation, the decisions (M1 design 6.1).

scripts/prepare_data.py does the I/O: the parquet shards, the tokenizer, the
files. What decides the contents of a token file lives here and is testable
without a network: which documents qualify, how a document becomes pieces,
which side of the split it lands on, and the order heldout.bin is written in.

  - A piece is PIECE_LEN = 4097 tokens, one document's, never two:
    TokenStream.batch reads seq_len + 1 tokens, so a 4096-token piece would take
    each document's last target from the next document.
  - A document is cut into whole pieces and the remainder dropped.
  - The split hashes the document's text (blake2b, never Python's salted
    hash()): stable across re-runs, and exact duplicates -- FineWeb is not
    deduplicated across dumps -- land on the same side.
  - heldout.bin is written in digest order, so any prefix of it is a random
    sample of documents. The in-run evals read its first 488 pieces.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .stream import PROVENANCE, TOKEN_DTYPE

PIECE_LEN = 4097
HELDOUT_PERCENT = 1
TRAIN, HELDOUT = "train.bin", "heldout.bin"
SPLIT_RULE = ("blake2b(text utf-8, digest_size=16) as a big-endian integer, "
              "mod 100 < 1: heldout, else train; heldout.bin in digest order")


def text_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def is_heldout(digest: bytes) -> bool:
    return int.from_bytes(digest, "big") % 100 < HELDOUT_PERCENT


def cut_pieces(ids: Sequence[int], piece_len: int = PIECE_LEN) -> np.ndarray:
    """[n, piece_len] whole pieces, in order; the remainder is dropped."""
    n = len(ids) // piece_len
    return np.asarray(ids[: n * piece_len], dtype=TOKEN_DTYPE).reshape(n, piece_len)


def yield_estimate(gpt2_counts: np.ndarray, ratio: float, prefilter: int,
                   piece_len: int = PIECE_LEN) -> Dict[str, int]:
    """What the long mode would get from documents of these GPT-2 lengths.

    FineWeb-edu's token_count column is GPT-2 tokens; ratio converts them to
    the training tokenizer's (measured by calibrate_ratio). An estimate: the
    ratio varies by document. Pieces are counted before the 1% held out.
    """
    counts = np.asarray(gpt2_counts, dtype=np.float64)
    kept = counts[counts >= prefilter]
    pieces = np.floor(np.floor(kept * ratio) / piece_len)
    return {"documents": int(counts.size), "prefiltered": int(kept.size),
            "qualifying": int((pieces > 0).sum()), "pieces": int(pieces.sum()),
            "tokens": int(pieces.sum()) * piece_len}


def calibrate_ratio(gpt2_counts: Sequence[int], counts: Sequence[int],
                    piece_len: int = PIECE_LEN, quantile: float = 0.999) -> Dict[str, Any]:
    """The training-tokenizer / GPT-2 token ratio over a sample of documents,
    and a loose pre-filter: the GPT-2 length below which a document cannot
    reach one piece unless its ratio is above the quantile-th of the sample.
    Rounded down to a multiple of 100."""
    gpt2 = np.asarray(gpt2_counts, dtype=np.float64)
    ours = np.asarray(counts, dtype=np.float64)
    per_document = ours / gpt2
    high = float(np.quantile(per_document, quantile))
    return {"documents": int(gpt2.size), "ratio": float(ours.sum() / gpt2.sum()),
            "ratio_quantile": quantile, "ratio_at_quantile": high,
            "prefilter": int(piece_len / high) // 100 * 100}


def prepare_long(batches: Iterable[Tuple[List[str], List[int]]],
                 encode: Callable[[List[str]], List[List[int]]],
                 prefilter: int, train_tokens: int, out: Path,
                 provenance: Dict[str, Any], piece_len: int = PIECE_LEN) -> Dict[str, Any]:
    """Write out/train.bin, out/heldout.bin and out/provenance.json.

    batches yields (texts, gpt2_counts) per parquet batch; encode tokenizes a
    list of texts without special tokens. Stops once train.bin holds at least
    train_tokens. provenance carries the source description and is completed
    with the counts. Returns it.
    """
    out.mkdir(parents=True, exist_ok=True)
    heldout: List[Tuple[bytes, np.ndarray]] = []
    docs = {"read": 0, "prefiltered": 0, "qualifying": 0, "train": 0, "heldout": 0}
    written = 0
    with open(out / TRAIN, "wb") as train:
        for texts, gpt2_counts in batches:
            docs["read"] += len(texts)
            chosen = [t for t, n in zip(texts, gpt2_counts) if n >= prefilter]
            docs["prefiltered"] += len(chosen)
            for text, ids in zip(chosen, encode(chosen) if chosen else []):
                pieces = cut_pieces(ids, piece_len)
                if not len(pieces):
                    continue
                docs["qualifying"] += 1
                digest = text_digest(text)
                if is_heldout(digest):
                    heldout.append((digest, pieces))
                    docs["heldout"] += 1
                    continue
                pieces.tofile(train)
                written += pieces.size
                docs["train"] += 1
                if written >= train_tokens:
                    break
            if written >= train_tokens:
                break
    held = 0
    with open(out / HELDOUT, "wb") as handle:
        for _, pieces in sorted(heldout, key=lambda item: item[0]):
            pieces.tofile(handle)
            held += pieces.size
    record = {**provenance, "piece_len": piece_len, "prefilter_gpt2_tokens": prefilter,
              "split": SPLIT_RULE, "documents": docs,
              "tokens": {"train": written, "heldout": held},
              "dtype": np.dtype(TOKEN_DTYPE).name,
              "bytes": {TRAIN: (out / TRAIN).stat().st_size,
                        HELDOUT: (out / HELDOUT).stat().st_size}}
    (out / PROVENANCE).write_text(json.dumps(record, indent=2) + "\n")
    return record
