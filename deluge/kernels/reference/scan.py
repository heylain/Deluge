"""Reference scans for the linear recurrence of spec 3.

    h_t = a_t * h_{t-1} + b_t * v_t,   a_t = exp(-Delta_t * lambda_t)

Two implementations, both pure PyTorch:

  sequential_scan  a loop over time. Obviously correct, and the ground truth
                   everything else is measured against.
  chunked_scan     the algorithm the Triton kernel will implement -- parallel
                   within a chunk, sequential across chunks. Same arithmetic,
                   different order, so a parity test against the sequential
                   version catches algorithm bugs before any kernel is written.

Neither is fast. That is the point: `docs/project-structure.md` principle 2 says
a Triton kernel may only exist next to a parity test against its reference, and
a reference that is clever is no longer a reference.

Everything here takes **log a**, never a. Spec 3 is explicit that lambda lives in
the log domain and a is materialised in fp32 inside the kernel: at a = 0.999 the
difference between 0.999 and 0.998 is a halving of the time constant, and that
is below fp16's resolution near 1.
"""

import torch

__all__ = ["input_scale", "sequential_scan", "chunked_scan"]


def input_scale(log_a: torch.Tensor) -> torch.Tensor:
    """b = sqrt(1 - a^2), the Griffin normalisation that bounds the state.

    Computed as sqrt(-expm1(2 log a)) rather than sqrt(1 - exp(2 log a)^2). The
    two are the same number and not the same float: near a = 1, which is exactly
    the regime spec 3 cares about, 1 - exp(x) for small x cancels almost every
    significant bit, while expm1 is accurate there by construction. At
    a = 0.999 the naive form has already lost about three decimal digits.
    """
    return torch.sqrt(torch.clamp(-torch.expm1(2 * log_a.float()), min=0.0))


def sequential_scan(log_a: torch.Tensor, bv: torch.Tensor,
                    h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The recurrence, one token at a time.

    log_a, bv: [batch, time, d_inner]; h0: [batch, d_inner].
    Returns (h for every t, h after the last t).
    """
    _check(log_a, bv, h0)
    a = torch.exp(log_a.float())
    bv = bv.float()
    h = h0.float()

    states = []
    for t in range(log_a.shape[1]):
        h = a[:, t] * h + bv[:, t]
        states.append(h)
    return torch.stack(states, dim=1), h


def chunked_scan(log_a: torch.Tensor, bv: torch.Tensor, h0: torch.Tensor,
                 chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The same recurrence, parallel within chunks of `chunk` tokens.

    Within a chunk, with L_t the cumulative sum of log a up to t,

        h_t = exp(L_t) * h_prev + sum_{s <= t} exp(L_t - L_s) * bv_s

    The decay between two positions is formed as a *difference of logs*, never
    as a ratio exp(L_t) / exp(L_s). They are algebraically identical; the ratio
    is not usable, because exp(L_s) underflows to zero a few hundred tokens into
    a chunk at a = 0.9 and the division then produces infinities. Since
    log a <= 0, every exp(L_t - L_s) here is in (0, 1].

    This materialises the chunk x chunk decay matrix, which the Triton kernel
    will hold in registers one tile at a time. Memory is O(batch * chunk^2 *
    d_inner) -- fine for a test, deliberately unfit for a training run.
    """
    _check(log_a, bv, h0)
    if chunk < 1:
        raise ValueError(f"chunk must be at least 1, got {chunk}")

    log_a, bv = log_a.float(), bv.float()
    time = log_a.shape[1]
    h_prev = h0.float()

    outputs = []
    for start in range(0, time, chunk):
        stop = min(start + chunk, time)
        log_a_c, bv_c = log_a[:, start:stop], bv[:, start:stop]
        width = stop - start

        cumulative = torch.cumsum(log_a_c, dim=1)                  # [B, C, D]
        # decay[b, t, s, d] = exp(L_t - L_s), the decay applied to bv_s by time t
        decay = torch.exp(cumulative.unsqueeze(2) - cumulative.unsqueeze(1))
        causal = torch.tril(torch.ones(width, width, dtype=torch.bool,
                                       device=log_a.device))
        decay = decay * causal[None, :, :, None]

        carried = torch.exp(cumulative) * h_prev.unsqueeze(1)
        within = (decay * bv_c.unsqueeze(1)).sum(dim=2)
        states = carried + within

        outputs.append(states)
        h_prev = states[:, -1]

    return torch.cat(outputs, dim=1), h_prev


def _check(log_a: torch.Tensor, bv: torch.Tensor, h0: torch.Tensor) -> None:
    if log_a.shape != bv.shape:
        raise ValueError(f"log_a {tuple(log_a.shape)} and bv {tuple(bv.shape)} "
                         f"must have the same shape")
    if log_a.ndim != 3:
        raise ValueError(f"expected [batch, time, d_inner], got {tuple(log_a.shape)}")
    if h0.shape != (log_a.shape[0], log_a.shape[2]):
        raise ValueError(f"h0 {tuple(h0.shape)} must be "
                         f"[batch, d_inner] = {(log_a.shape[0], log_a.shape[2])}")
    if (log_a > 0).any():
        raise ValueError("log_a must be <= 0; a = exp(log_a) is a decay in (0, 1]")
