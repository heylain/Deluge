"""Configuration dataclasses and invariant validation.

Importable without torch: this module is loaded by the trainer, the inference
engine and the eval harness alike, and by tooling that only wants parameter
counts.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import yaml


CFC_PER_UNIT = 3   # the (C, C, C, A) repeating unit of spec 2
CONV_KERNEL = 4    # DWConv_k4; the inference conv buffer holds CONV_KERNEL - 1


class ConfigError(ValueError):
    """A config violates an invariant that the model or kernels assume."""


@dataclass(frozen=True)
class ParamCount:
    """Parameter counts derived from a config, never from the spec tables."""

    embedding: int
    cfc: int
    attention: int
    ffn: int

    @property
    def total(self) -> int:
        return self.embedding + self.cfc + self.attention + self.ffn


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_inner: int
    n_decay_heads: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    window: int
    chunk: int
    summary_sources: Tuple[int, ...]
    vocab_size: int
    tie_embeddings: bool
    n_units: int
    d_ff: int
    n_sink: int

    def __post_init__(self) -> None:
        if self.d_inner % self.n_decay_heads:
            raise ConfigError(
                f"d_inner ({self.d_inner}) must be divisible by n_decay_heads "
                f"({self.n_decay_heads}): lambda is per-head and broadcasts over "
                f"d_inner // n_decay_heads channels"
            )
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
        if not self.summary_sources:
            raise ConfigError("summary_sources must name at least one CfC layer")
        if any(offset >= 0 for offset in self.summary_sources):
            raise ConfigError(
                f"summary_sources {self.summary_sources} must be negative offsets "
                f"relative to the attention layer: a state at offset >= 0 has not "
                f"been computed yet"
            )

    @property
    def channels_per_decay_head(self) -> int:
        """Channels each lambda broadcasts over; 1 means per-channel decay."""
        return self.d_inner // self.n_decay_heads

    @property
    def summary_input_dim(self) -> int:
        """Width W_sk/W_sv read: residual stream plus one CfC state per source."""
        return self.d_model + len(self.summary_sources) * self.d_inner

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

    def _cfc_layer_params(self) -> int:
        d, di, h = self.d_model, self.d_inner, self.n_decay_heads
        return (
            d * di                      # W_v
            + CONV_KERNEL * di          # depthwise conv
            + d * h + h                 # W_f, b_f
            + d * di                    # W_g
            + di * d                    # W_out
            + di                        # learned h0
            + (CONV_KERNEL - 1) * di    # learned initial conv buffer
            + di                        # RMSNorm over h
            + d                         # pre-norm
        )

    def _attention_layer_params(self) -> int:
        d, kv = self.d_model, self.n_kv_heads * self.head_dim
        q = self.n_heads * self.head_dim
        return (
            d * q                            # W_q
            + 2 * d * kv                     # W_k, W_v
            + q * d                          # W_o
            + 2 * self.summary_input_dim * kv  # W_sk, W_sv
            + 2 * self.n_sink * kv           # learned sink K/V
            + d                              # pre-norm
        )

    def _ffn_params(self) -> int:
        return 3 * self.d_model * self.d_ff + self.d_model

    def params(self) -> ParamCount:
        """Derive parameter counts from this config."""
        embedding = self.vocab_size * self.d_model
        if not self.tie_embeddings:
            embedding *= 2
        return ParamCount(
            embedding=embedding,
            cfc=self.n_cfc_layers * self._cfc_layer_params(),
            attention=self.n_attention_layers * self._attention_layer_params(),
            ffn=self.n_blocks * self._ffn_params(),
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
    data = _resolve(Path(path))

    known = {f.name for f in fields(ModelConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"unknown config key(s) {', '.join(unknown)}; "
            f"known keys are {', '.join(sorted(known))}"
        )
    missing = sorted(known - set(data))
    if missing:
        raise ConfigError(f"missing config key(s): {', '.join(missing)}")

    if "summary_sources" in data:
        data["summary_sources"] = tuple(data["summary_sources"])
    return ModelConfig(**data)
