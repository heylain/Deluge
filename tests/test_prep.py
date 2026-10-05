"""Long-document data prep (M1 design 6.1), with a fake tokenizer and no network."""

import hashlib
import json

import numpy as np
import pytest

from deluge.data import TokenStream, piece_batches
from deluge.data.prep import (PIECE_LEN, calibrate_ratio, cut_pieces, is_heldout,
                              prepare_long, text_digest, yield_estimate)

SMALL = 9          # a tiny piece length, so test documents stay short


def encode(texts):
    """One token per character: a document of n characters is n tokens."""
    return [[ord(c) % 1000 for c in text] for text in texts]


def documents(n, length, seed=0):
    rng = np.random.default_rng(seed)
    return ["".join(chr(97 + x) for x in rng.integers(0, 26, size=length)) for _ in range(n)]


def run(tmp_path, texts, prefilter=0, train_tokens=10**9, counts=None):
    counts = counts if counts is not None else [len(t) for t in texts]
    batches = [(texts[i:i + 7], counts[i:i + 7]) for i in range(0, len(texts), 7)]
    return prepare_long(batches, encode, prefilter, train_tokens, tmp_path,
                        {"dataset": "fake"}, piece_len=SMALL)


def test_the_piece_is_one_sequence_of_4096_plus_its_target():
    assert PIECE_LEN == 4096 + 1


def test_the_split_hashes_the_text_with_blake2b():
    digest = text_digest("a document")
    assert digest == hashlib.blake2b(b"a document", digest_size=16).digest()
    assert is_heldout(digest) == (int.from_bytes(digest, "big") % 100 < 1)


def test_about_one_percent_is_held_out():
    held = sum(is_heldout(text_digest(f"doc {i}")) for i in range(20_000))
    assert 140 <= held <= 260


def test_cut_pieces_drops_the_remainder_and_short_documents():
    assert cut_pieces(list(range(20)), 9).tolist() == [list(range(9)), list(range(9, 18))]
    assert cut_pieces(list(range(8)), 9).shape == (0, 9)
    assert cut_pieces(list(range(9)), 9).shape == (1, 9)     # exactly one piece


def test_duplicates_land_on_the_same_side(tmp_path):
    texts = documents(300, 20) * 2
    record = run(tmp_path, texts)
    assert record["documents"]["heldout"] % 2 == 0
    assert record["documents"]["train"] % 2 == 0


def test_heldout_is_written_in_digest_order(tmp_path):
    texts = documents(3000, 20)
    run(tmp_path, texts)
    held = sorted((text_digest(t), t) for t in texts if is_heldout(text_digest(t)))
    expected = np.concatenate([cut_pieces(encode([t])[0], SMALL).ravel() for _, t in held])
    assert np.array_equal(np.fromfile(tmp_path / "heldout.bin", dtype=np.uint16), expected)


def test_the_output_reads_back_in_piece_mode(tmp_path):
    record = run(tmp_path, documents(500, 30))
    stream = TokenStream.from_path(tmp_path / "train.bin", seq_len=SMALL - 1, seed=0)
    assert stream.n_sequences * SMALL == record["tokens"]["train"]
    heldout = np.fromfile(tmp_path / "heldout.bin", dtype=np.uint16)
    assert sum(len(i) for i, _ in piece_batches(heldout, SMALL, None, 8)) * SMALL == len(heldout)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["piece_len"] == SMALL and provenance["dataset"] == "fake"
    assert provenance["bytes"]["train.bin"] == 2 * record["tokens"]["train"]


def test_documents_under_one_piece_are_counted_and_dropped(tmp_path):
    # Review focus: the GPT-2 pre-filter is loose, so a document can pass it
    # and still tokenize to less than one piece.
    texts = documents(10, 5) + documents(10, 9, seed=1)
    record = run(tmp_path, texts)
    assert record["documents"]["prefiltered"] == 20
    assert record["documents"]["qualifying"] == 10


def test_the_prefilter_keeps_short_documents_from_the_tokenizer(tmp_path):
    seen = []

    def spy(texts):
        seen.extend(texts)
        return encode(texts)

    texts = documents(10, 30)
    prepare_long([(texts, [100] * 5 + [10] * 5)], spy, 50, 10**9, tmp_path,
                 {}, piece_len=SMALL)
    assert seen == texts[:5]


def test_stops_once_train_holds_the_target(tmp_path):
    record = run(tmp_path, documents(1000, 30), train_tokens=900)
    assert 900 <= record["tokens"]["train"] < 900 + 30
    assert record["documents"]["read"] < 1000


def test_the_yield_estimate_counts_whole_converted_pieces():
    estimate = yield_estimate(np.array([1000, 4000, 9000]), ratio=1.1, prefilter=3000)
    # 4000 * 1.1 = 4400 -> 1 piece; 9000 * 1.1 = 9900 -> 2 pieces.
    assert estimate == {"documents": 3, "prefiltered": 2, "qualifying": 2,
                        "pieces": 3, "tokens": 3 * PIECE_LEN}


def test_calibration_sets_a_loose_prefilter():
    gpt2 = [1000, 2000, 3000, 4000]
    ours = [1100, 2200, 3300, 6000]           # the last document runs at 1.5
    result = calibrate_ratio(gpt2, ours, quantile=1.0)
    assert result["ratio"] == pytest.approx(12600 / 10000)
    assert result["prefilter"] == int(4097 / 1.5) // 100 * 100 == 2700
