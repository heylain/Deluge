from pathlib import Path

import pytest
import yaml

from deluge.config import ConfigError, ModelConfig, load_model_config

# The dev config of spec 2.1, as a baseline each test perturbs by one field.
# Lives here, not in the dataclass: configs are data (configs/*.yaml), so
# ModelConfig deliberately has no defaults to drift out of sync with them.
REPO = Path(__file__).resolve().parents[1]

DEV = dict(
    d_model=768,
    d_inner=1152,
    n_decay_heads=12,
    n_heads=12,
    n_kv_heads=4,
    head_dim=64,
    window=512,
    chunk=64,
    summary_sources=(-1,),
    vocab_size=32000,
    tie_embeddings=True,
    n_units=3,
    d_ff=2048,
    n_sink=4,
)


def make_config(**overrides) -> ModelConfig:
    return ModelConfig(**{**DEV, **overrides})


def test_accepts_the_dev_config():
    assert make_config().channels_per_decay_head == 96


def test_rejects_d_inner_not_divisible_by_n_decay_heads():
    # lambda is per-head and broadcasts over d_inner/H channels; a ragged split
    # would silently mis-broadcast in the scan kernel.
    with pytest.raises(ConfigError, match="d_inner"):
        make_config(d_inner=1000)


def test_per_channel_decay_is_the_h_equals_d_inner_case():
    # ADR-0001 / spec 16: per-channel lambda is not a separate code path, it is
    # one decay head per channel.
    assert make_config(n_decay_heads=1152).channels_per_decay_head == 1


def test_rejects_window_not_a_multiple_of_chunk():
    # Spec 5: a chunk is summarised once it falls fully outside the window. If W
    # is not a whole number of chunks the raw/summary boundary drifts and the
    # train and inference masks disagree.
    with pytest.raises(ConfigError, match="window"):
        make_config(window=500)


def test_rejects_n_heads_not_divisible_by_n_kv_heads():
    # GQA: each KV head is shared by n_heads/n_kv_heads query heads.
    with pytest.raises(ConfigError, match="n_kv_heads"):
        make_config(n_heads=12, n_kv_heads=5)


def test_rejects_head_dim_that_does_not_reconstruct_d_model():
    # The output projection reads n_heads*head_dim and writes d_model.
    with pytest.raises(ConfigError, match="head_dim"):
        make_config(n_heads=12, head_dim=63)


def test_summary_projection_width_follows_the_source_list():
    # Spec 16 M0: summaries read the residual stream plus one designated CfC
    # state per source, so "every CfC layer" is a config change, not a rewrite.
    assert make_config(summary_sources=(-1,)).summary_input_dim == 768 + 1152
    assert make_config(summary_sources=(-1, -2, -3)).summary_input_dim == 768 + 3 * 1152


def test_rejects_summary_source_that_is_not_a_preceding_layer():
    # Offsets are relative to the attention layer; 0 or positive would read a
    # state that does not exist yet at this point in the stack.
    with pytest.raises(ConfigError, match="summary_sources"):
        make_config(summary_sources=(0,))


def test_tied_embeddings_are_counted_once():
    tied = make_config(tie_embeddings=True).params()
    untied = make_config(tie_embeddings=False).params()
    assert untied.total - tied.total == 32000 * 768


def test_trunk_params_scale_linearly_in_units():
    # Doubling the units doubles everything except the embedding, which is
    # shared. Catches per-layer terms accidentally counted once, and global
    # terms accidentally counted per layer.
    p3 = make_config(n_units=3).params()
    p6 = make_config(n_units=6).params()
    assert p3.embedding == p6.embedding
    assert p6.total - p3.total == p3.total - p3.embedding


def test_loads_a_model_config_from_yaml(tmp_path):
    path = tmp_path / "dev.yaml"
    path.write_text(yaml.safe_dump({k: list(v) if isinstance(v, tuple) else v
                                    for k, v in DEV.items()}))
    assert load_model_config(path).d_model == 768


def test_extends_overrides_only_the_named_keys(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({k: list(v) if isinstance(v, tuple) else v
                                    for k, v in DEV.items()}))
    child = tmp_path / "wide_ffn.yaml"
    child.write_text(yaml.safe_dump({"extends": "base.yaml", "d_ff": 4096}))

    cfg = load_model_config(child)
    assert cfg.d_ff == 4096
    assert cfg.d_model == 768


def test_rejects_an_unknown_key_in_yaml(tmp_path):
    path = tmp_path / "typo.yaml"
    data = {k: list(v) if isinstance(v, tuple) else v for k, v in DEV.items()}
    data["d_modell"] = 1024
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ConfigError, match="d_modell"):
        load_model_config(path)


# These two keep spec 2.1 honest: the table is derived from config, not the other
# way round, so a config change that moves the counts fails here until the spec
# is updated to match.

def test_dev_config_matches_the_param_count_in_the_spec():
    total = load_model_config(REPO / "configs/model/dev.yaml").params().total
    assert 112e6 < total < 114e6, f"spec 2.1 says ~113M, config gives {total/1e6:.1f}M"


def test_target_v1_config_matches_the_param_count_in_the_spec():
    total = load_model_config(REPO / "configs/model/target_v1.yaml").params().total
    assert 1.32e9 < total < 1.34e9, f"spec 2.1 says 1.331B, config gives {total/1e9:.3f}B"


def test_rejects_a_config_missing_a_required_key(tmp_path):
    path = tmp_path / "partial.yaml"
    data = {k: list(v) if isinstance(v, tuple) else v for k, v in DEV.items()}
    del data["window"]
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ConfigError, match="window"):
        load_model_config(path)


def test_rejects_a_circular_extends_chain(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"extends": "b.yaml"}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"extends": "a.yaml"}))

    with pytest.raises(ConfigError, match="circular"):
        load_model_config(tmp_path / "a.yaml")
