"""Corpus handling. See docs/project-structure.md for what else lands here
(tokenizer wrapper, domain mixture, chunk-aligned packing)."""

from .stream import TOKEN_DTYPE, TokenStream, permute

__all__ = ["TOKEN_DTYPE", "TokenStream", "permute"]
