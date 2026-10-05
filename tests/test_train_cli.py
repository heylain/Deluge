"""The parts of train.py that do not need torch: factory resolution and CLI.

Worth separating, because these are exactly the failures a Kaggle session hits
in its first ten seconds -- a typo in a path, a model factory that is not there
yet -- and they should be legible without a GPU in the room.
"""

import pytest

from deluge.config import load_model_config
from deluge.train.trainer import main, resolve_factory


def test_resolves_a_module_attr_factory():
    assert resolve_factory("deluge.config:load_model_config") is load_model_config


def test_rejects_a_spec_without_an_attribute():
    with pytest.raises(ValueError, match="module:attr"):
        resolve_factory("deluge.model")


def test_the_default_factory_is_the_model():
    pytest.importorskip("torch")
    from deluge.model import build
    assert resolve_factory("deluge.model:build") is build


def test_a_missing_module_names_the_flag():
    with pytest.raises(ImportError, match="--model-impl"):
        resolve_factory("deluge.no_such_module:build")


def test_missing_attribute_names_it():
    with pytest.raises(ImportError, match="no_such_factory"):
        resolve_factory("deluge.train:no_such_factory")


def test_cli_requires_the_four_paths():
    with pytest.raises(SystemExit):
        main([])


def test_cli_rejects_an_unknown_flag():
    with pytest.raises(SystemExit):
        main(["--model", "m", "--train", "t", "--data", "d", "--out", "o",
              "--epochs", "3"])
