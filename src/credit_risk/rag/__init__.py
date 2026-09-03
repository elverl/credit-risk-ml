"""Retrieval preparation utilities for the current knowledge base."""

from .chunking import build_chunks, resolve_expected_chunk_ids

__all__ = ["build_chunks", "resolve_expected_chunk_ids"]
