"""The data-prep command lines, against fakes of the Hub: no network."""

import ast
import fnmatch
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import kaggle_prep  # noqa: E402
import prepare_data  # noqa: E402

DATASET_SHA, TOKENIZER_SHA = "d" * 40, "t" * 40
COMMIT = "0123456789abcdef" * 2 + "01234567"
PIECE = 4097


class Hub:
    """huggingface_hub's HfApi and HfFileSystem and tokenizers' Tokenizer, in
    one fake. Shards are real parquet files on disk; every call is recorded."""

    def __init__(self, root: Path, shards):
        import pyarrow as pa
        import pyarrow.parquet as pq

        root.mkdir()
        self.files, self.calls, self.opens = {}, [], []
        for name, (texts, counts) in shards.items():
            local = root / name
            pq.write_table(pa.table({"text": texts, "token_count": counts}), local,
                           row_group_size=3)
            self.files[f"datasets/fake/ds@{DATASET_SHA}/sample/{name}"] = local

    def dataset_info(self, repo_id, revision=None):
        self.calls.append(("dataset_info", repo_id, revision))
        return SimpleNamespace(sha=DATASET_SHA)

    def model_info(self, repo_id, revision=None):
        self.calls.append(("model_info", repo_id, revision))
        return SimpleNamespace(sha=TOKENIZER_SHA)

    def glob(self, pattern):
        self.calls.append(("glob", pattern))
        return sorted(p for p in self.files if fnmatch.fnmatch(p, pattern))

    def open(self, path, mode="rb", **options):
        self.opens.append((path, options))
        return open(self.files[path], mode)

    def from_pretrained(self, identifier, revision="main"):
        self.calls.append(("from_pretrained", identifier, revision))
        return SimpleNamespace(
            get_vocab_size=lambda: 32000,
            # One token per character, as in test_prep.py.
            encode_batch=lambda texts, add_special_tokens: [
                SimpleNamespace(ids=[ord(c) % 1000 for c in t]) for t in texts])


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """hub(shards) installs a Hub serving {name: (texts, token_counts)}."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("pyarrow")

    def install(shards):
        fake = Hub(tmp_path / "shards", shards)
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)
        monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda: fake)
        monkeypatch.setattr(tokenizers, "Tokenizer",
                            SimpleNamespace(from_pretrained=fake.from_pretrained))
        return fake
    return install


def long_documents(n, seed=0):
    """n documents of one whole piece each, as (texts, GPT-2 token counts)."""
    rng = np.random.default_rng(seed)
    texts = ["".join(chr(97 + x) for x in rng.integers(0, 26, size=PIECE + 100))
             for _ in range(n)]
    return texts, [len(t) for t in texts]


def long_args(out, *extra):
    return ["long", "--dataset", "fake/ds", "--files", "sample/*.parquet",
            "--tokenizer", "fake/tok", "--prefilter", "0", "--out", str(out), *extra]


def test_prepare_data_has_the_three_modes(capsys):
    with pytest.raises(SystemExit) as exit_:
        prepare_data.main(["--help"])
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    assert all(mode in out for mode in ("measure", "long", "stream"))


def test_the_long_mode_needs_a_prefilter_and_a_target():
    with pytest.raises(SystemExit):
        prepare_data.main(["long", "--dataset", "d", "--files", "f", "--tokenizer", "t",
                           "--out", "o"])


def test_any_dataset_revision_is_resolved_to_a_sha(hub):
    fake = hub({"a.parquet": long_documents(2)})
    _, revision, paths = prepare_data.source("fake/ds", "sample/*.parquet", "main")
    assert revision == DATASET_SHA
    assert ("dataset_info", "fake/ds", "main") in fake.calls
    assert ("glob", f"datasets/fake/ds@{DATASET_SHA}/sample/*.parquet") in fake.calls
    assert len(paths) == 1


def test_the_tokenizer_is_loaded_at_a_resolved_sha(hub):
    import tokenizers

    fake = hub({"a.parquet": long_documents(2)})
    record, encode = prepare_data.tokenizer("fake/tok", "main")
    assert ("model_info", "fake/tok", "main") in fake.calls
    assert ("from_pretrained", "fake/tok", TOKENIZER_SHA) in fake.calls
    assert record == {"tokenizer": "fake/tok", "tokenizer_revision": TOKENIZER_SHA,
                      "tokenizers_version": tokenizers.__version__, "vocab_size": 32000}
    assert encode(["ab"]) == [[97, 98]]


def test_measure_prints_both_shas(hub, capsys):
    hub({"a.parquet": long_documents(4), "b.parquet": long_documents(3, seed=1)})
    assert prepare_data.main(["measure", "--dataset", "fake/ds", "--files", "sample/*.parquet",
                              "--revision", "main", "--tokenizer", "fake/tok",
                              "--tokenizer-revision", "main"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["revision"] == DATASET_SHA
    assert printed["tokenizer_revision"] == TOKENIZER_SHA
    assert printed["documents"] == 7


def test_measure_reads_token_counts_without_read_ahead(hub):
    # fsspec's default cache reads ahead 5 MiB per read: on FineWeb-edu, whose
    # row groups sit ~2.8 MB apart, the token_count pass would pull the text
    # too. Only the calibration pass, which reads text, keeps the default.
    fake = hub({"a.parquet": long_documents(4), "b.parquet": long_documents(3, seed=1)})
    prepare_data.main(["measure", "--dataset", "fake/ds", "--files", "sample/*.parquet",
                       "--tokenizer", "fake/tok"])
    shards = sorted(fake.files)
    counted = sorted(path for path, options in fake.opens if options == {"cache_type": "none"})
    default = [path for path, options in fake.opens if options == {}]
    assert counted == shards
    assert default == shards[:1]
    assert len(fake.opens) == len(counted) + len(default)


def test_token_counts_come_back_in_path_order(hub):
    fake = hub({"a.parquet": (["x"] * 4, [1, 2, 3, 4]), "b.parquet": (["y"] * 3, [5, 6, 7])})
    counts = prepare_data.token_counts(fake, sorted(fake.files))
    assert counts.tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_long_pins_what_the_data_depends_on(hub, tmp_path):
    import tokenizers

    fake = hub({"a.parquet": long_documents(6), "b.parquet": long_documents(6, seed=1)})
    out = tmp_path / "out"
    assert prepare_data.main(long_args(out, "--revision", "main", "--tokenizer-revision", "main",
                                       "--train-tokens", str(3 * PIECE),
                                       "--builder-commit", "c" * 40)) == 0
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["revision"] == DATASET_SHA
    assert provenance["tokenizer_revision"] == TOKENIZER_SHA
    assert provenance["tokenizers_version"] == tokenizers.__version__
    assert provenance["builder_commit"] == "c" * 40
    assert provenance["train_tokens_target"] == 3 * PIECE
    assert provenance["tokens"]["train"] == 3 * PIECE
    assert ("from_pretrained", "fake/tok", TOKENIZER_SHA) in fake.calls


def test_long_defaults_the_builder_commit_to_this_checkouts_head(hub, tmp_path):
    hub({"a.parquet": long_documents(6)})
    out = tmp_path / "out"
    prepare_data.main(long_args(out, "--train-tokens", str(PIECE)))
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert json.loads((out / "provenance.json").read_text())["builder_commit"] == head


def test_the_builder_commit_is_null_outside_a_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert prepare_data.head_commit(tmp_path) is None


def test_a_short_long_build_exits_non_zero_with_the_fix(hub, tmp_path):
    hub({"a.parquet": long_documents(6)})
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exit_:
        prepare_data.main(long_args(out, "--train-tokens", "1000000000"))
    message = exit_.value.code
    assert isinstance(message, str)           # SystemExit(str): status 1
    assert "1,000,000,000" in message and "--files" in message
    assert not {"train.bin", "heldout.bin", "provenance.json"} & {p.name for p in out.iterdir()}


def test_the_prep_kernel_is_a_private_cpu_kernel_with_internet(tmp_path):
    kaggle_prep.main(["--owner", "heylaine", "--commit", COMMIT, "--dir", str(tmp_path),
                      "--", "--dataset", "HuggingFaceFW/fineweb-edu", "--prefilter", "2700"])
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "heylaine/deluge-prep-long4k" and meta["is_private"] is True
    assert meta["enable_gpu"] is False and meta["enable_internet"] is True
    code = (tmp_path / "prep_kernel.py").read_text()
    ast.parse(code)
    params = json.loads(ast.literal_eval(code.split("json.loads(")[1].split(")\n")[0]))
    assert params["commit"] == COMMIT
    assert params["args"] == ["--dataset", "HuggingFaceFW/fineweb-edu", "--prefilter", "2700"]
    assert '"--out", "/kaggle/working"' in code


def test_the_prep_kernel_owns_the_output_directory(tmp_path):
    with pytest.raises(ValueError, match="--out"):
        kaggle_prep.render(tmp_path, "o", COMMIT, ["--out", "/elsewhere"])


def test_the_prep_kernel_owns_the_builder_commit(tmp_path):
    with pytest.raises(ValueError, match="--builder-commit"):
        kaggle_prep.render(tmp_path, "o", COMMIT, ["--builder-commit", "other"])


@pytest.mark.parametrize("commit", ["abc1234", "HEAD", COMMIT.upper(), COMMIT + "0"])
def test_the_prep_kernel_takes_only_a_full_commit_sha(tmp_path, commit):
    # The kernel records --commit as provenance's builder_commit.
    with pytest.raises(ValueError, match="40-hex"):
        kaggle_prep.render(tmp_path, "o", commit, ["--dataset", "d"])


def test_the_prep_kernel_pins_its_installs_and_records_its_commit(tmp_path, monkeypatch):
    kaggle_prep.render(tmp_path, "heylaine", COMMIT, ["--dataset", "d"])
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda command, **_: ran.append(command))
    exec((tmp_path / "prep_kernel.py").read_text(), {"__name__": "__main__"})
    pip, prep = ran[2], ran[3]
    assert pip[-3:] == ["huggingface_hub==1.33.0", "pyarrow==25.0.1", "tokenizers==0.23.2"]
    assert prep[prep.index("--builder-commit") + 1] == COMMIT
    assert prep[-2:] == ["--out", "/kaggle/working"]
