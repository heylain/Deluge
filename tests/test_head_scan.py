"""The head-batched scan must equal the sequential reference, forwards and backwards.

head_chunked_scan is the scan every CfC and LRU layer trains on in M1, so this
file is to it what test_scan_parity.py is to the reference chunked scan. The
reference takes log a per channel; per-head log a is expanded to it with
repeat_interleave, which is exactly the broadcast spec 3 describes.
"""

import pytest
import torch

from deluge.kernels.head_scan import head_chunked_scan
from deluge.kernels.reference.scan import sequential_scan

# Spec 3's init range: a in [0.9, 0.999] at Delta=1.
LOG_A_MIN, LOG_A_MAX = -0.10536, -0.001
HEADS, PER_HEAD = 3, 4


def make(batch=2, time=100, seed=0, log_a_min=LOG_A_MIN, log_a_max=LOG_A_MAX,
         requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    span = log_a_max - log_a_min
    log_a = log_a_min + span * torch.rand(batch, time, HEADS, generator=generator)
    bv = torch.randn(batch, time, HEADS * PER_HEAD, generator=generator)
    h0 = torch.randn(batch, HEADS * PER_HEAD, generator=generator)
    if requires_grad:
        for tensor in (log_a, bv, h0):
            tensor.requires_grad_(True)
    return log_a, bv, h0


def reference(log_a, bv, h0):
    return sequential_scan(log_a.repeat_interleave(PER_HEAD, dim=-1), bv, h0)


@pytest.mark.parametrize("chunk", [1, 16, 64])
@pytest.mark.parametrize("time", [1, 63, 64, 100, 128])
def test_matches_the_sequential_scan(chunk, time):
    # time 63, 100: a ragged final chunk, padded with log a = 0 and bv = 0.
    log_a, bv, h0 = make(time=time)
    expected, expected_last = reference(log_a, bv, h0)
    got, got_last = head_chunked_scan(log_a, bv, h0, chunk)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_last, expected_last, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("chunk", [16, 64])
def test_gradients_match_the_sequential_scan(chunk):
    def grads(scan, **kwargs):
        log_a, bv, h0 = make(requires_grad=True)
        states, last = scan(log_a, bv, h0, **kwargs)
        weights = torch.arange(1, states.shape[1] + 1)[None, :, None]
        ((states * weights).sum() + last.sum()).backward()
        return log_a.grad, bv.grad, h0.grad

    for expected, got in zip(grads(reference), grads(head_chunked_scan, chunk=chunk)):
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


def test_long_chunks_do_not_underflow():
    # exp(L_s) is below float32's smallest normal a few hundred tokens into a
    # chunk at a = 0.9; the difference-of-logs form never forms it.
    log_a, bv, h0 = make(time=512, log_a_min=-0.105, log_a_max=-0.105)
    states, last = head_chunked_scan(log_a, bv, h0, 512)
    assert torch.isfinite(states).all() and torch.isfinite(last).all()
    torch.testing.assert_close(states, reference(log_a, bv, h0)[0], rtol=1e-4, atol=1e-5)


def test_fast_decay_stays_finite_forwards_and_backwards():
    # lambda = 5 across a 64-token chunk: the above-diagonal exponent reaches
    # 5 * 63 = 315, far past exp's float32 overflow at ~88.7.
    def run(scan, **kwargs):
        log_a, bv, h0 = make(time=128, log_a_min=-5.0, log_a_max=-5.0, requires_grad=True)
        states, last = scan(log_a, bv, h0, **kwargs)
        (states.sum() + last.sum()).backward()
        return states.detach(), log_a.grad, bv.grad, h0.grad

    for expected, got in zip(run(reference), run(head_chunked_scan, chunk=64)):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


def test_the_last_state_is_the_last_row():
    log_a, bv, h0 = make(time=100)
    states, last = head_chunked_scan(log_a, bv, h0, 64)
    assert torch.equal(states[:, -1], last)


def test_runs_in_fp32_under_autocast():
    # Spec 3: a is materialised in fp32. A bf16 cumulative sum near a = 1
    # would lose the time constant the model is learning.
    log_a, bv, h0 = make()
    expected, _ = head_chunked_scan(log_a, bv, h0, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got, _ = head_chunked_scan(log_a, bv, h0, 16)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, expected)


def test_rejects_a_width_that_does_not_split_into_heads():
    log_a, bv, h0 = make()
    with pytest.raises(ValueError, match="channels per head"):
        head_chunked_scan(log_a, bv[..., :-1], h0[..., :-1], 16)


def test_rejects_a_non_positive_chunk():
    log_a, bv, h0 = make()
    with pytest.raises(ValueError, match="chunk"):
        head_chunked_scan(log_a, bv, h0, 0)
