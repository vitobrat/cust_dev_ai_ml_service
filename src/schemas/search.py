"""Shared search result schemas used across search domain and services."""

import uuid
from typing import Any

from pydantic import BaseModel


class SearchResultItem(BaseModel):
    """Single search result with a relevance score.

    Attributes:
        id: Unique point identifier in Qdrant.
        score: Cosine similarity score (higher = more relevant).
        payload: Document metadata stored alongside the embedding.
    """

    id: uuid.UUID
    score: float
    payload: dict[str, Any]
