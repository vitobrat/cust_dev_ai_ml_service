"""Request and response schemas for the search domain."""

import uuid

from pydantic import BaseModel, Field

from src.schemas.api_base import ResponseBase
from src.schemas.search import SearchResultItem


class PostSearchRequest(BaseModel):
    """Request body for the semantic search endpoint.

    Attributes:
        user_id: Owner of the document space to search within.
        query: Natural language query text.
        top_k: Maximum number of results to return.
    """

    user_id: uuid.UUID
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class PostSearchResponse(ResponseBase):
    """Response body for the semantic search endpoint.

    Attributes:
        results: Ranked list of matching documents, ordered by relevance.
    """

    msg: list[SearchResultItem]
