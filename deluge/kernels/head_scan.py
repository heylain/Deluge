"""Head-batched chunked scan: the training scan of M1, in plain PyTorch.

    h_t = a_t * h_{t-1} + bv_t,   a_t = exp(log_a_t), one log_a per decay head

kernels/reference/scan.py takes log a per channel and materialises a
[B, C, C, D] decay tensor per chunk. That is right for a reference and unfit
for training at 4096 tokens. Spec 3 makes lambda per head, so within a chunk
one [C, C] decay matrix serves all P channels of a head, and the intra-chunk
work for the whole sequence is one batched matmul (SSD-style):

  1. intra-chunk, every chunk at once: decay[b, h, n, t, s] = exp(L_t - L_s)
     for s <= t, with L the within-chunk cumulative sum of log a, and
     within = decay @ bv
  2. inter-chunk: a loop over the N chunks carries the chunk-end state forward
  3. combine: states = within + exp(L_t) * h_chunk_start

Same numerics as the reference: fp32 with autocast off, and decay formed as a
difference of logs, never a ratio. tests/test_head_scan.py holds it to
sequential_scan forwards and backwards. A Triton kernel replaces it at M3
(M1 design D2); until then it is the only training scan, and the mixers import
it directly -- the kernel registry arrives with a second implementation.
"""

import torch
import torch.nn.functional as F

__all__ = ["head_chunked_scan"]


def head_chunked_scan(log_a: torch.Tensor, bv: torch.Tensor, h0: torch.Tensor,
                      chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The recurrence with per-head decay, parallel within chunks.

    log_a: [batch, time, heads], <= 0. bv: [batch, time, width] with width a
    whole number of channels per head, laid out head-major (head j owns
    channels j*P .. (j+1)*P - 1). h0: [batch, width].
    Returns (states [batch, time, width], h after the last t [batch, width]),
    both fp32.

    log_a <= 0 is the caller's invariant (log_a = -Delta * softplus(...)) and
    is not checked here: the check is a GPU sync per layer per step.
    """
    _check(log_a, bv, h0, chunk)
    batch, time, heads = log_a.shape
    width = bv.shape[-1]
    per_head = width // heads

    with torch.autocast(device_type=log_a.device.type, enabled=False):
        log_a, bv, h = log_a.float(), bv.float(), h0.float()
        pad = (-time) % chunk
        if pad:
            # log a = 0 and bv = 0 leave the state where it is, so padding the
            # ragged final chunk cannot move any real position.
            log_a = F.pad(log_a, (0, 0, 0, pad))
            bv = F.pad(bv, (0, 0, 0, pad))
        n = (time + pad) // chunk

        la = log_a.reshape(batch, n, chunk, heads).permute(0, 3, 1, 2)            # [B,H,N,C]
        x = bv.reshape(batch, n, chunk, heads, per_head).permute(0, 3, 1, 2, 4)   # [B,H,N,C,P]
        cumulative = torch.cumsum(la, dim=-1)                                     # L_t
        exponent = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)           # [B,H,N,C,C]
        causal = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=la.device))
        # -inf before the exp, never a mask after it: above the diagonal the
        # exponent is positive and overflows (the reference scan's old NaN).
        decay = torch.exp(exponent.masked_fill(~causal, float("-inf")))
        within = decay @ x                                                        # [B,H,N,C,P]

        carry = torch.exp(cumulative[..., -1])                                    # [B,H,N]
        state = h.reshape(batch, heads, per_head)
        starts = []
        for j in range(n):
            starts.append(state)
            state = carry[:, :, j, None] * state + within[:, :, j, -1]
        start = torch.stack(starts, dim=2)                                        # [B,H,N,P]

        states = within + torch.exp(cumulative).unsqueeze(-1) * start.unsqueeze(3)
        states = states.permute(0, 2, 3, 1, 4).reshape(batch, n * chunk, width)[:, :time]
    return states, states[:, -1]


def _check(log_a: torch.Tensor, bv: torch.Tensor, h0: torch.Tensor, chunk: int) -> None:
    if log_a.ndim != 3 or bv.ndim != 3:
        raise ValueError(f"expected log_a [batch, time, heads] and bv [batch, time, "
                         f"width], got {tuple(log_a.shape)} and {tuple(bv.shape)}")
    if log_a.shape[:2] != bv.shape[:2]:
        raise ValueError(f"log_a {tuple(log_a.shape)} and bv {tuple(bv.shape)} must "
                         f"agree on batch and time")
    heads, width = log_a.shape[2], bv.shape[2]
    if width % heads:
        raise ValueError(f"bv width ({width}) must be a whole number of channels per "
                         f"head ({heads} heads)")
    if h0.shape != (bv.shape[0], width):
        raise ValueError(f"h0 {tuple(h0.shape)} must be [batch, width] = "
                         f"{(bv.shape[0], width)}")
    if chunk < 1:
        raise ValueError(f"chunk must be at least 1, got {chunk}")
