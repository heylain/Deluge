"""The chunked scan must equal the sequential one, forwards and backwards.

The sequential scan is the definition of the recurrence; the chunked scan is the
algorithm the Triton kernel will implement. Everything expensive later -- the
kernel, its backward pass, the fused decode step -- is checked against these, so
if the two here disagree the whole chain is measuring the wrong thing.

Backward matters as much as forward. A chunked forward with a subtly wrong
gradient trains to something, just not to what the sequential version would, and
nothing in a loss curve says so.
"""

import pytest
import torch

from deluge.kernels.reference.scan import chunked_scan, input_scale, sequential_scan

# Spec 3's init range: a in [0.9, 0.999] at Delta=1, so log a in [-0.105, -0.001].
LOG_A_MIN, LOG_A_MAX = -0.10536, -0.001


def make(batch=2, time=37, width=8, seed=0, log_a_min=LOG_A_MIN,
         log_a_max=LOG_A_MAX, requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    span = log_a_max - log_a_min
    log_a = log_a_min + span * torch.rand(batch, time, width, generator=generator)
    bv = torch.randn(batch, time, width, generator=generator)
    h0 = torch.randn(batch, width, generator=generator)
    if requires_grad:
        for tensor in (log_a, bv, h0):
            tensor.requires_grad_(True)
    return log_a, bv, h0


@pytest.mark.parametrize("chunk", [1, 2, 4, 8, 16, 37, 64])
def test_chunked_matches_sequential(chunk):
    log_a, bv, h0 = make()
    expected, expected_last = sequential_scan(log_a, bv, h0)
    got, got_last = chunked_scan(log_a, bv, h0, chunk)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_last, expected_last, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("time", [1, 2, 63, 64, 65, 128])
def test_a_ragged_final_chunk_is_handled(time):
    # Sequence length is not a multiple of the chunk in general, and the last
    # partial chunk is where an off-by-one in the causal mask hides.
    log_a, bv, h0 = make(time=time)
    expected, _ = sequential_scan(log_a, bv, h0)
    got, _ = chunked_scan(log_a, bv, h0, 64)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_chunk_of_one_is_exactly_the_sequential_scan():
    log_a, bv, h0 = make()
    expected, _ = sequential_scan(log_a, bv, h0)
    got, _ = chunked_scan(log_a, bv, h0, 1)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("chunk", [4, 16])
def test_gradients_match_the_sequential_scan(chunk):
    def grads(scan, **kwargs):
        log_a, bv, h0 = make(requires_grad=True)
        states, _ = scan(log_a, bv, h0, **kwargs)
        (states * torch.arange(1, states.shape[1] + 1)[None, :, None]).sum().backward()
        return log_a.grad, bv.grad, h0.grad

    for reference, candidate in zip(grads(sequential_scan),
                                    grads(chunked_scan, chunk=chunk)):
        torch.testing.assert_close(candidate, reference, rtol=1e-4, atol=1e-5)


def test_the_last_state_is_the_last_row_of_the_states():
    log_a, bv, h0 = make()
    states, last = chunked_scan(log_a, bv, h0, 8)
    assert torch.equal(states[:, -1], last)


# --------------------------------------------------------------------------- #
# The recurrence's two limits
# --------------------------------------------------------------------------- #

def test_no_decay_accumulates_every_input():
    # a = 1: h_t is h0 plus the running sum. The scan has no business losing
    # anything here, and a wrong cumulative-sum direction shows up immediately.
    log_a = torch.zeros(1, 5, 3)
    bv = torch.ones(1, 5, 3)
    h0 = torch.zeros(1, 3)
    states, _ = chunked_scan(log_a, bv, h0, 2)
    torch.testing.assert_close(states[0, :, 0], torch.arange(1.0, 6.0))


def test_strong_decay_forgets_the_initial_state():
    log_a = torch.full((1, 64, 2), -5.0)
    bv = torch.zeros(1, 64, 2)
    h0 = torch.full((1, 2), 1e6)
    _, last = chunked_scan(log_a, bv, h0, 16)
    assert last.abs().max() < 1e-6


def test_long_chunks_do_not_underflow():
    # The ratio form exp(L_t)/exp(L_s) dies here: at a = 0.9, exp(L_s) is under
    # float32's smallest normal a few hundred tokens in, and the division makes
    # infinities. The difference-of-logs form is bounded by construction.
    log_a, bv, h0 = make(time=512, log_a_min=-0.105, log_a_max=-0.105)
    states, last = chunked_scan(log_a, bv, h0, 512)
    assert torch.isfinite(states).all() and torch.isfinite(last).all()
    expected, _ = sequential_scan(log_a, bv, h0)
    torch.testing.assert_close(states, expected, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- #
# input_scale
# --------------------------------------------------------------------------- #

def test_input_scale_is_accurate_where_the_naive_form_is_not():
    # Spec 3: precision near a = 1 is the whole ballgame. sqrt(1 - exp(2x))
    # cancels almost every significant bit for small x; expm1 does not.
    log_a = torch.tensor([-1e-7, -1e-6, -1e-5], dtype=torch.float32)
    exact = torch.sqrt(-torch.expm1(2 * log_a.double()))

    naive = torch.sqrt(torch.clamp(1 - torch.exp(2 * log_a), min=0.0))
    naive_error = ((naive.double() - exact) / exact).abs().max()
    ours_error = ((input_scale(log_a).double() - exact) / exact).abs().max()

    assert ours_error < 1e-6, f"expm1 form is off by {ours_error:.2e}"
    assert naive_error > 1e-3, (
        "the naive form got accurate; if so this guard can go, but check that "
        "the test is still exercising small |log a|")


def test_input_scale_is_bounded_by_one():
    # b = sqrt(1 - a^2) with a in (0, 1] is the Griffin normalisation that keeps
    # the state bounded; b > 1 would mean it does not.
    log_a = torch.linspace(-10, 0, 50).reshape(1, -1, 1)
    scale = input_scale(log_a)
    assert (scale >= 0).all() and (scale <= 1).all()


def test_no_decay_means_no_input():
    # a = 1 pairs with b = 0: a state that never forgets never listens either.
    assert input_scale(torch.zeros(1, 1, 1)).item() == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

def test_rejects_a_positive_log_a():
    # a > 1 is a growing state, not a decay. Spec 3's A6 note about negative
    # eigenvalues is about the sign of a, not about |a| > 1.
    log_a, bv, h0 = make()
    with pytest.raises(ValueError, match="log_a must be <= 0"):
        sequential_scan(log_a + 1.0, bv, h0)


def test_rejects_mismatched_shapes():
    log_a, bv, h0 = make()
    with pytest.raises(ValueError, match="same shape"):
        chunked_scan(log_a, bv[:, :-1], h0, 4)
    with pytest.raises(ValueError, match="h0"):
        chunked_scan(log_a, bv, h0[:, :-1], 4)


def test_rejects_a_non_positive_chunk():
    log_a, bv, h0 = make()
    with pytest.raises(ValueError, match="chunk"):
        chunked_scan(log_a, bv, h0, 0)
