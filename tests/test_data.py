"""The token stream's ordering guarantees.

Everything here is really one claim: order is a pure function of (seed, step),
so a resumed run can jump straight to step N without replaying anything.
"""

import numpy as np
import pytest

from deluge.data import TokenStream, permute, piece_batches

CORPUS = np.arange(10_000, dtype=np.uint16)


def stream(seq_len=64, seed=3, tokens=CORPUS):
    return TokenStream(tokens, seq_len=seq_len, seed=seed)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 16, 17, 64, 65, 255, 1000, 4096, 4097])
def test_permute_visits_every_index_exactly_once(n):
    # An epoch that skipped or repeated sequences would quietly change the data
    # distribution, and nothing downstream would notice.
    assert sorted(permute(i, n, key=12345) for i in range(n)) == list(range(n))


def test_permute_is_a_shuffle_not_the_identity():
    n = 4096
    moved = sum(permute(i, n, key=7) != i for i in range(n))
    assert moved > 0.9 * n


def test_different_keys_give_different_orders():
    n = 1024
    a = [permute(i, n, key=1) for i in range(n)]
    b = [permute(i, n, key=2) for i in range(n)]
    assert a != b


def test_an_epoch_covers_the_corpus_exactly_once():
    s = stream()
    starts = [s.start_of(i) for i in range(s.n_sequences)]
    assert sorted(starts) == [i * s.seq_len for i in range(s.n_sequences)]


def test_the_next_epoch_reshuffles():
    s = stream()
    first = [s.start_of(i) for i in range(s.n_sequences)]
    second = [s.start_of(i + s.n_sequences) for i in range(s.n_sequences)]
    assert sorted(first) == sorted(second)
    assert first != second, "epoch 2 replays epoch 1 in the same order"


def test_position_does_not_depend_on_how_it_was_reached():
    # The whole point: step 500 is the same whether you trained to it or not.
    fresh = stream()
    walked = stream()
    for i in range(500):
        walked.start_of(i)
    assert fresh.start_of(500) == walked.start_of(500)


def test_targets_are_inputs_shifted_by_one():
    s = stream()
    inputs, targets = s.batch(s.indices_for_step(0, 8))
    assert np.array_equal(inputs[:, 1:], targets[:, :-1])


def test_sequences_never_run_off_the_end():
    # n_sequences carries a -1 so the last sequence still has a target token.
    s = stream(seq_len=64)
    last = s.start_of(s.n_sequences - 1)
    assert last + s.seq_len + 1 <= len(s.tokens)


def test_steps_do_not_overlap():
    s = stream()
    a = set(s.indices_for_step(0, 8).tolist())
    b = set(s.indices_for_step(1, 8).tolist())
    assert not a & b


def test_seed_changes_the_order():
    a = [stream(seed=1).start_of(i) for i in range(20)]
    b = [stream(seed=2).start_of(i) for i in range(20)]
    assert a != b


def test_rejects_a_corpus_too_short_for_one_sequence():
    with pytest.raises(ValueError, match="no sequence"):
        TokenStream(np.arange(30, dtype=np.uint16), seq_len=64, seed=0)


def test_reads_a_memmapped_file(tmp_path):
    path = tmp_path / "tokens.bin"
    CORPUS.tofile(path)
    s = TokenStream.from_path(path, seq_len=64, seed=3)
    assert s.n_sequences == stream().n_sequences
    assert np.array_equal(s.batch(s.indices_for_step(0, 4))[0],
                          stream().batch(stream().indices_for_step(0, 4))[0])


# --------------------------------------------------------------------------- #
# Piece mode (M1 design 6.2): whole documents cut into pieces of seq_len + 1
# --------------------------------------------------------------------------- #

PIECE = 65                                   # seq_len 64 + the shifted target


def pieces(n=10):
    return np.arange(n * PIECE, dtype=np.uint16)


def test_piece_mode_keeps_every_target_inside_its_piece():
    s = TokenStream(pieces(), seq_len=64, seed=3, piece_len=PIECE)
    for i in range(s.n_sequences):
        inputs, targets = s.batch(np.array([i]))
        start = s.start_of(i)
        assert start % PIECE == 0
        assert inputs[0, 0] == start and targets[0, -1] == start + PIECE - 1


def test_piece_mode_reads_every_piece_once_per_epoch():
    s = TokenStream(pieces(), seq_len=64, seed=3, piece_len=PIECE)
    assert s.n_sequences == 10
    assert sorted(s.start_of(i) for i in range(10)) == [i * PIECE for i in range(10)]


def test_a_partial_final_piece_is_never_read():
    # A truncated train.bin must not yield a sequence that runs off its end.
    s = TokenStream(pieces()[:-7], seq_len=64, seed=3, piece_len=PIECE)
    assert s.n_sequences == 9
    assert max(s.start_of(i) for i in range(9)) + PIECE <= len(s.tokens)


def test_rejects_pieces_that_do_not_fit_the_sequence_length():
    with pytest.raises(ValueError, match="pieces of 65"):
        TokenStream(pieces(), seq_len=32, seed=3, piece_len=PIECE)


def test_from_path_reads_the_piece_length_from_provenance(tmp_path):
    path = tmp_path / "train.bin"
    pieces().tofile(path)
    (tmp_path / "provenance.json").write_text('{"piece_len": 65}')
    assert TokenStream.from_path(path, seq_len=64, seed=3).n_sequences == 10


def test_without_provenance_the_stream_is_contiguous_as_before(tmp_path):
    path = tmp_path / "tokens.bin"
    pieces().tofile(path)
    s = TokenStream.from_path(path, seq_len=64, seed=3)
    assert s.piece_len is None and s.n_sequences == (650 - 1) // 64


def test_piece_batches_reads_pieces_in_file_order():
    batches = list(piece_batches(pieces(), PIECE, n_pieces=5, batch=2))
    assert [len(inputs) for inputs, _ in batches] == [2, 2, 1]
    inputs, targets = batches[2]
    assert inputs[0, 0] == 4 * PIECE and targets[0, -1] == 5 * PIECE - 1
    assert inputs.dtype == np.int64


def test_piece_batches_stops_at_the_last_whole_piece():
    assert sum(len(i) for i, _ in piece_batches(pieces()[:-1], PIECE, None, 4)) == 9
    assert sum(len(i) for i, _ in piece_batches(pieces(), PIECE, 50, 4)) == 10
