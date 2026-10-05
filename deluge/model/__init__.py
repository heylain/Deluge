"""The CfC-attention hybrid (spec 2-5): deluge.model:build is the factory
deluge.train uses by default."""

from .model import Model, build

__all__ = ["Model", "build"]
