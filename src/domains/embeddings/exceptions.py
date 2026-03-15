"""Embeddings domain exception hierarchy."""

from src.infrastructure.exceptions import EmbeddingError


class EmbeddingUpsertError(EmbeddingError):
    """Raised when text embedding upsert to Qdrant fails."""


class EmbeddingDeleteError(EmbeddingError):
    """Raised when deleting embeddings from Qdrant fails."""
