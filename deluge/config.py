"""Configuration dataclasses and invariant validation.

Importable without torch: this module is loaded by the trainer, the inference
engine and the eval harness alike, and by tooling that only wants parameter
counts.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

CFC_PER_UNIT = 3   # the (C, C, C, A) repeating unit of spec 2


class ConfigError(ValueError):
    """A config violates an invariant that the model or kernels assume."""


def _build(cls, data: Dict[str, Any], what: str):
    """Construct a dataclass from a mapping, rejecting unknown/missing keys."""
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"unknown {what} key(s) {', '.join(unknown)}; "
            f"known keys are {', '.join(sorted(known))}"
        )
    missing = sorted(known - set(data))
    if missing:
        raise ConfigError(f"missing {what} key(s): {', '.join(missing)}")
    return cls(**data)


# --------------------------------------------------------------------------- #
# Mixer blocks. Spec 11 A1 pits CfC-mix against the B3 conv and B4 LRU
# baselines; each implements the same interface so the swap is a config string.
# A mixer's conv buffer holds conv_kernel - 1 tokens at inference.
# --------------------------------------------------------------------------- #

def _check_decay_split(d_inner: int, n_decay_heads: int) -> None:
    if d_inner % n_decay_heads:
        raise ConfigError(
            f"d_inner ({d_inner}) must be divisible by n_decay_heads "
            f"({n_decay_heads}): lambda is per-head and broadcasts over "
            f"d_inner // n_decay_heads channels"
        )


@dataclass(frozen=True)
class CfCMixer:
    """Delta-aware gated recurrence. Gates are input-only (ADR-0001)."""

    d_inner: int
    n_decay_heads: int
    conv_kernel: int
    delta_aware: bool

    def __post_init__(self) -> None:
        _check_decay_split(self.d_inner, self.n_decay_heads)

    @property
    def channels_per_decay_head(self) -> int:
        """Channels each lambda broadcasts over; 1 means per-channel decay."""
        return self.d_inner // self.n_decay_heads

    @property
    def state_width(self) -> int:
        return self.d_inner

    @property
    def uses_delta(self) -> bool:
        """False is the A3 control arm: MoD gaps stop advancing Delta."""
        return self.delta_aware

    def params(self, d_model: int) -> int:
        d, di, h, k = d_model, self.d_inner, self.n_decay_heads, self.conv_kernel
        return (
            d * di              # W_v
            + k * di            # depthwise conv
            + d * h + h         # W_f, b_f
            + d * di            # W_g
            + di * d            # W_out
            + di                # learned h0
            + (k - 1) * di      # learned initial conv buffer
            + di                # RMSNorm over h
            + d                 # pre-norm
        )


@dataclass(frozen=True)
class GatedConvMixer:
    """B3: LFM2-style double-gated short conv. No recurrent state at all."""

    d_inner: int
    conv_kernel: int

    @property
    def state_width(self) -> int:
        return 0  # nothing for a chunk summary to read

    @property
    def uses_delta(self) -> bool:
        return False

    def params(self, d_model: int) -> int:
        d, di, k = d_model, self.d_inner, self.conv_kernel
        return (
            3 * d * di          # the two gates and the candidate
            + k * di            # depthwise conv
            + di * d            # W_out
            + (k - 1) * di      # learned initial conv buffer
            + d                 # pre-norm
        )


@dataclass(frozen=True)
class LRUMixer:
    """B4: plain LRU. Decay and state, but no conv and no Delta argument."""

    d_inner: int
    n_decay_heads: int

    def __post_init__(self) -> None:
        _check_decay_split(self.d_inner, self.n_decay_heads)

    @property
    def channels_per_decay_head(self) -> int:
        return self.d_inner // self.n_decay_heads

    @property
    def state_width(self) -> int:
        return self.d_inner

    @property
    def uses_delta(self) -> bool:
        return False

    def params(self, d_model: int) -> int:
        d, di, h = d_model, self.d_inner, self.n_decay_heads
        return (
            d * di              # W_v
            + d * h + h         # W_f, b_f
            + d * di            # W_g
            + di * d            # W_out
            + di                # learned h0
            + di                # RMSNorm over h
            + d                 # pre-norm
        )


MIXER_TYPES = {"cfc": CfCMixer, "gated_conv": GatedConvMixer, "lru": LRUMixer}
MixerConfig = Union[CfCMixer, GatedConvMixer, LRUMixer]


def _build_mixer(data: Dict[str, Any]) -> MixerConfig:
    spec = dict(data)
    kind = spec.pop("type", None)
    if kind not in MIXER_TYPES:
        raise ConfigError(
            f"mixer type {kind!r} is not one of {', '.join(sorted(MIXER_TYPES))}"
        )
    return _build(MIXER_TYPES[kind], spec, f"mixer.{kind}")


# --------------------------------------------------------------------------- #
# FFN blocks. Both expose params()/active_params() so spec 8's swap-in rule is
# a config change: v2 overrides `ffn:` and nothing else.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DenseFFN:
    """SwiGLU: gate, up, down."""

    d_ff: int

    def params(self, d_model: int) -> int:
        return 3 * d_model * self.d_ff + d_model  # + pre-norm

    def active_params(self, d_model: int) -> int:
        return self.params(d_model)



@dataclass(frozen=True)
class MoEFFN:
    """Fine-grained MoE: top_k routed experts plus one always-on shared expert."""

    n_experts: int
    d_ff_expert: int
    d_ff_shared: int
    top_k: int

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= self.n_experts:
            raise ConfigError(
                f"top_k ({self.top_k}) must be between 1 and n_experts "
                f"({self.n_experts})"
            )

    def _routed(self, d_model: int, n: int) -> int:
        return n * 3 * d_model * self.d_ff_expert

    def _always_on(self, d_model: int) -> int:
        return (
            3 * d_model * self.d_ff_shared        # shared expert
            + d_model * self.n_experts            # router
            + self.n_experts                      # per-expert load-balancing bias
            + d_model                             # pre-norm
        )

    def params(self, d_model: int) -> int:
        return self._routed(d_model, self.n_experts) + self._always_on(d_model)

    def active_params(self, d_model: int) -> int:
        """Only top_k experts and the shared expert are read per token."""
        return self._routed(d_model, self.top_k) + self._always_on(d_model)



FFN_TYPES = {"dense": DenseFFN, "moe": MoEFFN}
FFNConfig = Union[DenseFFN, MoEFFN]


def _build_ffn(data: Dict[str, Any]) -> FFNConfig:
    spec = dict(data)
    kind = spec.pop("type", None)
    if kind not in FFN_TYPES:
        raise ConfigError(
            f"ffn type {kind!r} is not one of {', '.join(sorted(FFN_TYPES))}"
        )
    return _build(FFN_TYPES[kind], spec, f"ffn.{kind}")


@dataclass(frozen=True)
class MoDConfig:
    """Mixture-of-Depths. Spec 6: applied to every Nth unit's CfC-mix + FFN.

    Attention blocks never skip in v1. The train-time top-k router is
    non-causal; `threshold` belongs to the causal predictor that replaces it at
    inference, and is the tier 0/1 speed dial of spec 14.
    """

    capacity: float        # rho: fraction of tokens processed per sequence
    every_n_units: int
    threshold: float       # causal predictor cutoff at inference

    def __post_init__(self) -> None:
        if not 0 < self.capacity <= 1:
            raise ConfigError(
                f"mod capacity ({self.capacity}) must be in (0, 1]; it is the "
                f"fraction of tokens the block processes"
            )
        if self.every_n_units < 1:
            raise ConfigError(
                f"mod every_n_units ({self.every_n_units}) must be at least 1"
            )
        if not 0 <= self.threshold <= 1:
            raise ConfigError(
                f"mod threshold ({self.threshold}) must be in [0, 1]: it is "
                f"compared against a sigmoid"
            )

    def params_per_gated_block(self, d_model: int) -> int:
        """Scalar router w_r, plus a linear->sigmoid causal predictor."""
        return d_model + (d_model + 1)


@dataclass(frozen=True)
class MTPConfig:
    """Multi-token prediction head. Spec 7, ablated at A4.

    DeepSeek-style: per depth, one extra mixer + FFN block reading
    W_p [RMSNorm(x_t^L); RMSNorm(emb(x_{t+1}))], sharing the trunk's embedding
    and unembedding. The head's mixer is whatever `mixer:` says, so an A1
    baseline arm gets an MTP head built from its own mixer rather than a CfC one.

    `share_trunk_block` is the M3 question of spec 16 -- whether the head reuses
    the last trunk unit's weights instead of carrying its own -- kept as a flag
    because it has to be answered at A4 on measured acceptance rate, before the
    target run's parameter budget is fixed, and because it is impossible to
    retrofit into trained weights.
    """

    depth: int
    share_trunk_block: bool

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ConfigError(
                f"mtp depth ({self.depth}) must be at least 1; omit the `mtp:` "
                f"block entirely for no MTP head (the A0-A3 rungs of spec 11)"
            )

    def params(self, d_model: int, block_params: int) -> int:
        """Per depth: the input projection and norms, plus a block unless shared.

        W_p reads a concatenation of two d_model vectors, so it is 2*d_model^2 --
        on target v1 that is 8.4M per depth before the block, which is why
        sharing is worth measuring rather than assuming.
        """
        per_depth = 2 * d_model * d_model + 2 * d_model
        if not self.share_trunk_block:
            per_depth += block_params
        return self.depth * per_depth


@dataclass(frozen=True)
class ParamCount:
    """Parameter counts derived from a config, never from the spec tables."""

    embedding: int
    cfc: int
    attention: int
    ffn: int
    ffn_active: int
    mod: int
    mtp: int
    mtp_active: int

    @property
    def total(self) -> int:
        return (self.embedding + self.cfc + self.attention + self.ffn
                + self.mod + self.mtp)

    @property
    def active_total(self) -> int:
        """Params read per token; differs from total only for MoE."""
        return (self.embedding + self.cfc + self.attention + self.ffn_active
                + self.mod + self.mtp_active)



@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    mixer: MixerConfig
    n_heads: int
    n_kv_heads: int
    head_dim: int
    window: int
    chunk: int
    summary_sources: Tuple[int, ...]
    vocab_size: int
    tie_embeddings: bool
    n_units: int
    n_sink: int
    ffn: FFNConfig
    mod: Optional[MoDConfig]
    mtp: Optional[MTPConfig]

    def __post_init__(self) -> None:
        if self.window % self.chunk:
            raise ConfigError(
                f"window ({self.window}) must be a whole number of chunks "
                f"({self.chunk}): the summary boundary in the attention mask is "
                f"defined by chunks falling fully outside the window"
            )
        if self.n_heads % self.n_kv_heads:
            raise ConfigError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads "
                f"({self.n_kv_heads}) for GQA"
            )
        if self.n_heads * self.head_dim != self.d_model:
            raise ConfigError(
                f"n_heads * head_dim ({self.n_heads} * {self.head_dim} = "
                f"{self.n_heads * self.head_dim}) must equal d_model "
                f"({self.d_model})"
            )
        # An empty list is legal and means window-only attention: the A0 and A1
        # rungs of spec 11, where A2 is exactly the diff that turns summaries on.
        if any(offset >= 0 for offset in self.summary_sources):
            raise ConfigError(
                f"summary_sources {self.summary_sources} must be negative offsets "
                f"relative to the attention layer: a state at offset >= 0 has not "
                f"been computed yet"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        data = dict(data)
        if "mixer" in data:
            data["mixer"] = _build_mixer(data["mixer"])
        if "ffn" in data:
            data["ffn"] = _build_ffn(data["ffn"])
        if data.get("mod") is not None:
            data["mod"] = _build(MoDConfig, data["mod"], "mod")
        if data.get("mtp") is not None:
            data["mtp"] = _build(MTPConfig, data["mtp"], "mtp")
        if "summary_sources" in data:
            data["summary_sources"] = tuple(data["summary_sources"])
        return _build(cls, data, "config")

    # ---- derived shapes -------------------------------------------------- #

    @property
    def uses_summaries(self) -> bool:
        """False is window-only attention -- A0/A1; A2 is the diff that flips it."""
        return bool(self.summary_sources)

    @property
    def summary_input_dim(self) -> int:
        """Width W_sk/W_sv read: residual stream plus one CfC state per source.

        Zero when summaries are off, because then there is no W_sk/W_sv at all --
        not d_model with nothing to project.
        """
        if not self.uses_summaries:
            return 0
        return self.d_model + len(self.summary_sources) * self.mixer.state_width

    @property
    def gated_unit_indices(self) -> Tuple[int, ...]:
        """Units whose CfC-mix and FFN blocks are MoD-gated.

        Counting starts at the second unit: the first unit is never gated,
        since that is where features are still forming.
        """
        if self.mod is None:
            return ()
        step = self.mod.every_n_units
        return tuple(range(step - 1, self.n_units, step))

    @property
    def n_gated_cfc_blocks(self) -> int:
        return len(self.gated_unit_indices) * CFC_PER_UNIT

    @property
    def n_cfc_layers(self) -> int:
        return CFC_PER_UNIT * self.n_units

    @property
    def n_attention_layers(self) -> int:
        return self.n_units

    @property
    def n_blocks(self) -> int:
        """Every mixer block is followed by its own FFN."""
        return self.n_cfc_layers + self.n_attention_layers

    # ---- parameter accounting -------------------------------------------- #

    def _attention_layer_params(self) -> int:
        d, kv = self.d_model, self.n_kv_heads * self.head_dim
        q = self.n_heads * self.head_dim
        return (
            d * q                              # W_q
            + 2 * d * kv                       # W_k, W_v
            + q * d                            # W_o
            + 2 * self.summary_input_dim * kv  # W_sk, W_sv; 0 when window-only
            + 2 * self.n_sink * kv             # learned sink K/V
            + d                                # pre-norm
        )

    def _mtp_params(self, ffn_params: int) -> int:
        """MTP head params against a trunk block costed with the given FFN.

        Called twice for MoE -- once with the FFN's total and once with its
        active count -- because the head's FFN is the same block type as the
        trunk's and is routed the same way.
        """
        if self.mtp is None:
            return 0
        block = self.mixer.params(self.d_model) + ffn_params
        return self.mtp.params(self.d_model, block)

    def params(self) -> ParamCount:
        """Derive parameter counts from this config."""
        embedding = self.vocab_size * self.d_model
        if not self.tie_embeddings:
            embedding *= 2
        return ParamCount(
            embedding=embedding,
            cfc=self.n_cfc_layers * self.mixer.params(self.d_model),
            attention=self.n_attention_layers * self._attention_layer_params(),
            ffn=self.n_blocks * self.ffn.params(self.d_model),
            ffn_active=self.n_blocks * self.ffn.active_params(self.d_model),
            mod=(
                0 if self.mod is None
                else self.n_gated_cfc_blocks
                * self.mod.params_per_gated_block(self.d_model)
            ),
            mtp=self._mtp_params(self.ffn.params(self.d_model)),
            mtp_active=self._mtp_params(self.ffn.active_params(self.d_model)),
        )


def _resolve(path: Path, _seen: Tuple[Path, ...] = ()) -> Dict[str, Any]:
    """Load a YAML file, applying `extends:` chains relative to each file."""
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ConfigError(f"circular extends: {chain}")

    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("extends", None)
    if parent is None:
        return data
    merged = _resolve(path.parent / parent, (*_seen, path))
    merged.update(data)
    return merged


def load_model_config(path: Union[str, Path]) -> ModelConfig:
    """Build a ModelConfig from a YAML file, following `extends:`."""
    return ModelConfig.from_dict(_resolve(Path(path)))
