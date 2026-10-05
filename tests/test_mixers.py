"""The slot mixers: state continuity, the spec's equations, and the M1 init.

Each mixer is checked against a per-token implementation written straight from
its equations, so the scan (tested in test_head_scan.py) and the wiring around
it -- conv tap order, the b scaling, the gate -- are tested separately.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from deluge.config import CfCMixer, GatedConvMixer, LRUMixer
from deluge.model.init import GATE_BIAS, decay_rates
from deluge.model.mixers import CfC, GatedConv, LRU
from deluge.model.mixers.base import causal_depthwise_conv
from deluge.model.state import MixerState

D = 16
TINY_CFC = CfCMixer(d_inner=24, n_decay_heads=3, conv_kernel=4, delta_aware=True,
                    a_min=0.9, a_max=0.999)
TINY_LRU = LRUMixer(d_inner=24, n_decay_heads=3, a_min=0.9, a_max=0.999)
TINY_CONV = GatedConvMixer(d_inner=24, conv_kernel=3)
DEV_CFC = CfCMixer(d_inner=1152, n_decay_heads=12, conv_kernel=4, delta_aware=True,
                   a_min=0.9, a_max=0.999)

MIXERS = {
    "cfc": lambda: CfC(D, TINY_CFC),
    "lru": lambda: LRU(D, TINY_LRU),
    "gated_conv": lambda: GatedConv(D, TINY_CONV),
}


def perturbed(make):
    """A mixer moved off its init, so zero h0, zero buffers and W_f = 0 cannot
    hide a state bug."""
    torch.manual_seed(0)
    module = make()
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return module


# ---- state ----------------------------------------------------------------- #

@pytest.mark.parametrize("kind", MIXERS)
@pytest.mark.parametrize("split", [1, 37, 64, 99])
def test_a_split_sequence_continues_exactly(kind, split):
    # 37 splits mid scan-chunk, 64 on a chunk edge.
    module = perturbed(MIXERS[kind])
    x = torch.randn(2, 100, D)
    whole, _ = module(x)
    first, state = module(x[:, :split])
    second, _ = module(x[:, split:], state)
    torch.testing.assert_close(torch.cat([first, second], dim=1), whole,
                               rtol=1e-5, atol=1e-5)


def test_a_fresh_sequence_starts_from_the_learned_state():
    # state=None means h0 and conv_buf0, broadcast over the batch -- not zeros.
    module = perturbed(MIXERS["cfc"])
    x = torch.randn(2, 5, D)
    explicit = MixerState(h=module.h0.detach().expand(2, -1).clone(),
                          conv_buf=module.conv_buf0.detach().expand(2, -1, -1).clone(),
                          gap=torch.zeros(2))
    torch.testing.assert_close(module(x)[0], module(x, explicit)[0])


def test_delta_of_one_is_the_default():
    module = perturbed(MIXERS["cfc"])
    x = torch.randn(2, 10, D)
    torch.testing.assert_close(module(x, delta=torch.ones(2, 10))[0], module(x)[0])


@pytest.mark.parametrize("kind", ["lru", "gated_conv"])
def test_the_baselines_refuse_a_delta(kind):
    with pytest.raises(ValueError, match="Delta"):
        MIXERS[kind]()(torch.randn(1, 4, D), delta=torch.ones(1, 4))


# ---- the equations --------------------------------------------------------- #

def cfc_by_hand(m, x, conv=True):
    """Spec 3, one token at a time, from the module's weights. conv=False is
    the LRU."""
    batch, time, _ = x.shape
    per_head = m.W_v.out_features // m.n_decay_heads
    u = x @ m.W_v.weight.T
    if conv:
        k = m.conv.shape[-1]
        padded = torch.cat([m.conv_buf0.expand(batch, -1, -1), u], dim=1)
    h, ys = m.h0.expand(batch, -1), []
    for t in range(time):
        if conv:
            window = padded[:, t:t + k]                       # oldest input first
            v = F.silu((window * m.conv[:, 0, :].T).sum(dim=1))
        else:
            v = F.silu(u[:, t])
        lam = F.softplus(x[:, t] @ m.W_f.weight.T + m.W_f.bias)
        a = torch.exp(-lam).repeat_interleave(per_head, dim=-1)
        h = a * h + torch.sqrt(1 - a ** 2) * v
        gate = F.silu(x[:, t] @ m.W_g.weight.T + m.W_g.bias)
        ys.append((m.norm_h(h) * gate) @ m.W_out.weight.T)
    return torch.stack(ys, dim=1)


def gated_conv_by_hand(m, x):
    batch, time, _ = x.shape
    b_gate, c_gate, candidate = (x @ m.W_in.weight.T).chunk(3, dim=-1)
    k = m.conv.shape[-1]
    padded = torch.cat([m.conv_buf0.expand(batch, -1, -1), b_gate * candidate], dim=1)
    ys = [(c_gate[:, t] * (padded[:, t:t + k] * m.conv[:, 0, :].T).sum(dim=1))
          @ m.W_out.weight.T for t in range(time)]
    return torch.stack(ys, dim=1)


@pytest.mark.parametrize("kind, by_hand", [
    ("cfc", cfc_by_hand),
    ("lru", lambda m, x: cfc_by_hand(m, x, conv=False)),
    ("gated_conv", gated_conv_by_hand),
])
def test_matches_its_equations_token_by_token(kind, by_hand):
    module = perturbed(MIXERS[kind])
    x = torch.randn(2, 70, D)
    with torch.no_grad():
        torch.testing.assert_close(module(x)[0], by_hand(module, x), rtol=1e-4, atol=1e-5)


# ---- init (M1 design 4.4) ---------------------------------------------------- #

def test_the_decay_init_is_the_geometric_grid_exactly():
    module = CfC(768, DEV_CFC)
    lam = F.softplus(module.W_f.bias.double())
    slow, fast = -math.log(0.999), -math.log(0.9)
    grid = [slow * (fast / slow) ** (i / 11) for i in range(12)]
    torch.testing.assert_close(torch.exp(-lam), torch.exp(-torch.tensor(grid, dtype=torch.float64)),
                               rtol=1e-6, atol=0.0)
    # Design D6: memories of 999 ... 9 tokens, half of them past 100.
    assert [round(1 / value) for value in decay_rates(0.9, 0.999, 12).tolist()] == [
        999, 655, 429, 281, 184, 120, 79, 52, 34, 22, 14, 9]
    assert torch.equal(module.W_f.weight, torch.zeros_like(module.W_f.weight))


def test_the_lru_starts_on_the_same_grid():
    lru = LRU(768, LRUMixer(d_inner=1152, n_decay_heads=12, a_min=0.9, a_max=0.999))
    assert torch.equal(lru.W_f.bias, CfC(768, DEV_CFC).W_f.bias)


@pytest.mark.parametrize("kind", ["cfc", "lru"])
def test_the_gate_starts_at_silu_of_its_bias(kind):
    module = MIXERS[kind]()
    gate = F.silu(module.W_g(torch.randn(3, 5, D)))
    torch.testing.assert_close(gate, torch.full_like(gate, F.silu(torch.tensor(GATE_BIAS)).item()))
    assert F.silu(torch.tensor(GATE_BIAS)).item() == pytest.approx(0.9995, abs=1e-4)


@pytest.mark.parametrize("kind", ["cfc", "gated_conv"])
def test_both_convs_start_as_the_identity(kind):
    module = MIXERS[kind]()
    taps = module.conv[:, 0, :]
    assert torch.equal(taps[:, -1], torch.ones(24))
    assert torch.equal(taps[:, :-1], torch.zeros(24, taps.shape[1] - 1))
    u = torch.randn(2, 7, 24)
    buffer = torch.zeros(2, taps.shape[1] - 1, 24)
    torch.testing.assert_close(causal_depthwise_conv(u, module.conv, buffer)[0], u)


def test_the_conv_buffer_carries_the_last_inputs():
    u = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    weight = torch.randn(3, 1, 4)
    _, buffer = causal_depthwise_conv(u, weight, torch.zeros(2, 3, 3))
    assert torch.equal(buffer, u[:, -3:])
