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


def test_missing_model_module_points_at_m1():
    # Until M1 lands there is no deluge.model:build (the package exists, the
    # factory does not), and the error should say so rather than reading as a
    # broken install.
    with pytest.raises(ImportError, match="M1"):
        resolve_factory("deluge.model:build")


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
