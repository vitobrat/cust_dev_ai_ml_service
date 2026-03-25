"""Shared search result schemas used across search domain and services."""

import uuid
from typing import Any

from pydantic import BaseModel, Field

from src.schemas.api_base import ResponseBase


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


class SearchRequest(BaseModel):
    """Request body for the semantic search endpoint.

    Attributes:
        user_id: Owner of the document space to search within.
        query: Natural language query text.
        top_k: Maximum number of results to return.
    """

    user_id: uuid.UUID
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResponse(ResponseBase):
    """Response body for the semantic search endpoint.

    Attributes:
        results: Ranked list of matching documents, ordered by relevance.
    """

    msg: list[SearchResultItem]
