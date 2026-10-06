"""Corpus handling. See docs/project-structure.md for what else lands here
(tokenizer wrapper, domain mixture, chunk-aligned packing)."""

from .stream import (HELDOUT, PROVENANCE, TOKEN_DTYPE, TRAIN, TokenStream,
                     load_provenance, permute, piece_batches)

__all__ = ["HELDOUT", "PROVENANCE", "TOKEN_DTYPE", "TRAIN", "TokenStream",
           "load_provenance", "permute", "piece_batches"]
