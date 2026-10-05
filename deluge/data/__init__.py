"""Corpus handling. See docs/project-structure.md for what else lands here
(tokenizer wrapper, domain mixture, chunk-aligned packing)."""

from .stream import (PROVENANCE, TOKEN_DTYPE, TokenStream, load_provenance,
                     permute, piece_batches)

__all__ = ["PROVENANCE", "TOKEN_DTYPE", "TokenStream", "load_provenance",
           "permute", "piece_batches"]
