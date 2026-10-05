"""Attention against a brute-force reference that shares no code with it.

M1 design D3 rests on this file: A0's transformer is our own Block stack with
attention in every slot, so its forward pass is only independent of the hybrid
if this test is. The reference therefore re-derives everything: RoPE in complex
form, KV heads expanded by explicit indexing, the mask as a list of visible
positions, and softmax(q k^T / sqrt(d)) v one query at a time, in float64.
"""

import math

import pytest
import torch

from deluge.config import ModelConfig
from deluge.model.mixers.attention import Attention, attention_mask
from deluge.model.rope import apply_rope, rope_angles

TIME = 24
TINY = dict(d_model=64, mixer={"type": "attention"}, n_heads=4, n_kv_heads=2,
            head_dim=16, window=8, chunk=8, summary_sources=[], vocab_size=97,
            tie_embeddings=True, n_units=1, n_sink=2,
            ffn={"type": "dense", "d_ff": 128}, mod=None, mtp=None)


def config(window: int) -> ModelConfig:
    return ModelConfig.from_dict({**TINY, "window": window})


def complex_rope(x: torch.Tensor, theta: float = 1e6) -> torch.Tensor:
    """RoPE as the paper writes it: channel pair j at position t is the complex
    number x_2j + i x_2j+1, multiplied by e^(i t theta^(-2j/d)). x: [T, d]."""
    time, d = x.shape
    freqs = theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    turn = torch.polar(torch.ones(time, d // 2, dtype=torch.float64),
                       torch.arange(time, dtype=torch.float64)[:, None] * freqs)
    z = torch.view_as_complex(x.double().reshape(time, d // 2, 2).contiguous())
    return torch.view_as_real(z * turn).reshape(time, d)


def brute_force(module: Attention, x: torch.Tensor) -> torch.Tensor:
    batch, time, _ = x.shape
    heads, kv_heads, d = module.n_heads, module.n_kv_heads, module.head_dim
    q = (x.double() @ module.W_q.weight.double().T).reshape(batch, time, heads, d)
    k = (x.double() @ module.W_k.weight.double().T).reshape(batch, time, kv_heads, d)
    v = (x.double() @ module.W_v.weight.double().T).reshape(batch, time, kv_heads, d)
    out = torch.zeros(batch, time, heads, d, dtype=torch.float64)
    for b in range(batch):
        for h in range(heads):
            g = h // (heads // kv_heads)                 # this query head's KV head
            q_h, k_h = complex_rope(q[b, :, h]), complex_rope(k[b, :, g])
            keys = torch.cat([module.sink_k[g].double(), k_h])        # sinks: no RoPE
            values = torch.cat([module.sink_v[g].double(), v[b, :, g]])
            for t in range(time):
                visible = list(range(module.n_sink)) + [
                    module.n_sink + s for s in range(time) if t - module.window < s <= t]
                scores = keys[visible] @ q_h[t] / math.sqrt(d)
                out[b, t, h] = torch.softmax(scores, dim=0) @ values[visible]
    return out.reshape(batch, time, heads * d) @ module.W_o.weight.double().T


@pytest.mark.parametrize("window", [8, 16, 24, 32])
def test_matches_the_brute_force_reference(window):
    # 8 and 16 are windows inside the sequence, 24 is exactly T, 32 is past it.
    torch.manual_seed(0)
    module = Attention(config(window))
    with torch.no_grad():                    # zero sinks would hide a sink bug
        module.sink_k.normal_()
        module.sink_v.normal_()
    x = torch.randn(2, TIME, 64)
    y, state = module(x)
    assert state is None
    torch.testing.assert_close(y.double(), brute_force(module, x), rtol=1e-4, atol=1e-5)


def test_rope_matches_the_complex_form():
    x = torch.randn(4096, 64)
    cos, sin = rope_angles(torch.arange(4096), 64)
    torch.testing.assert_close(apply_rope(x, cos, sin).double(), complex_rope(x),
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("t, s, shift", [(10, 3, 100), (4000, 3990, -3000), (7, 7, 2000)])
def test_rope_scores_depend_only_on_the_offset(t, s, shift):
    q, k = torch.randn(1, 64), torch.randn(1, 64)
    cos, sin = rope_angles(torch.arange(4096), 64)

    def score(t, s):
        return (apply_rope(q, cos[t:t + 1], sin[t:t + 1])
                @ apply_rope(k, cos[s:s + 1], sin[s:s + 1]).T)

    torch.testing.assert_close(score(t, s), score(t + shift, s + shift),
                               rtol=1e-4, atol=1e-4)


def test_the_mask_shows_the_sinks_always_and_the_window_band():
    mask = attention_mask(6, 2, 3, torch.device("cpu"))
    assert mask.shape == (6, 8) and mask[:, :2].all()
    assert mask[0, 2:].tolist() == [True, False, False, False, False, False]
    assert mask[5, 2:].tolist() == [False, False, False, True, True, True]


def test_attention_refuses_a_carried_state():
    module = Attention(config(8))
    _, state = module(torch.randn(1, 4, 64))
    with pytest.raises(ValueError, match="M4"):
        module(torch.randn(1, 4, 64), state=object())
