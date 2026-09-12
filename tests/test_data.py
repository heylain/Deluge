"""The token stream's ordering guarantees.

Everything here is really one claim: order is a pure function of (seed, step),
so a resumed run can jump straight to step N without replaying anything.
"""

import numpy as np
import pytest

from deluge.data import TokenStream, permute

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
