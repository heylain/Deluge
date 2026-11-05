"""Eval accounting and the skip counter, without torch."""

import numpy as np
import pytest

from deluge.train.monitors import BUCKETS, EvalAccumulator, SkipCounter


def test_the_loss_is_the_mean_over_every_token():
    acc = EvalAccumulator()
    acc.add(np.full((2, 4096), 3.0), {})
    acc.add(np.full((1, 4096), 6.0), {})
    result = acc.result()
    assert result["loss"] == pytest.approx(4.0) and result["tokens"] == 3 * 4096


def test_position_buckets_split_the_sequence():
    losses = np.concatenate([np.full(512, 1.0), np.full(1024, 2.0), np.full(2560, 3.0)])[None]
    acc = EvalAccumulator()
    acc.add(losses, {})
    assert acc.result()["buckets"] == {"0-512": 1.0, "512-1536": 2.0, "1536-4096": 3.0}


def test_a_short_sequence_leaves_later_buckets_empty():
    acc = EvalAccumulator()
    acc.add(np.ones((2, 64)), {})
    assert acc.result()["buckets"] == {"0-512": 1.0, "512-1536": None, "1536-4096": None}


def test_the_decay_summary_is_a_at_the_mean_lambda_per_layer():
    # Two sequences, two layers, three heads. Mean lambda per head, then
    # a = exp(-mean lambda), then min / median / max over the heads.
    lam = np.array([[[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]],
                    [[0.3, 0.4, 0.5], [3.0, 3.0, 3.0]]])
    rms = np.array([[1.0, 2.0], [3.0, 4.0]])
    acc = EvalAccumulator()
    acc.add(np.ones((2, 8)), {"decay_lambda": lam, "state_rms": rms})
    result = acc.result()
    np.testing.assert_allclose(result["decay"]["min"], np.exp(-np.array([0.4, 2.0])))
    np.testing.assert_allclose(result["decay"]["median"], np.exp(-np.array([0.3, 2.0])))
    np.testing.assert_allclose(result["decay"]["max"], np.exp(-np.array([0.2, 2.0])))
    np.testing.assert_allclose(result["state_rms"], [2.0, 3.0])


def test_arms_without_recurrent_layers_report_no_decay():
    acc = EvalAccumulator()
    acc.add(np.ones((1, 8)), {})
    assert "decay" not in acc.result() and "state_rms" not in acc.result()


def test_an_empty_eval_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="no held-out tokens"):
        EvalAccumulator().result()


def test_the_buckets_cover_the_training_sequence():
    assert BUCKETS[0][0] == 0 and BUCKETS[-1][1] == 4096
    assert all(a[1] == b[0] for a, b in zip(BUCKETS, BUCKETS[1:]))


def test_a_skip_is_a_drop_in_the_loss_scale():
    counter = SkipCounter()
    counter.observe(65536.0, 32768.0)      # overflow: step skipped, scale halved
    counter.observe(32768.0, 32768.0)      # a normal step
    counter.observe(32768.0, 65536.0)      # growth after 2000 clean steps
    assert counter.count == 1


def test_the_skip_count_round_trips_through_a_checkpoint():
    counter = SkipCounter()
    counter.observe(2.0, 1.0)
    restored = SkipCounter()
    restored.load_state_dict(counter.state_dict())
    assert restored.count == 1
