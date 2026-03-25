"""Request and response schemas for the search domain."""

from src.schemas.search import SearchRequest, SearchResponse, SearchResultItem


class PostSearchRequest(SearchRequest):
    """Request body for the semantic search endpoint.

    Attributes:
        user_id: Owner of the document space to search within.
        query: Natural language query text.
        top_k: Maximum number of results to return.
    """


class PostSearchResponse(SearchResponse):
    """Response body for the semantic search endpoint.

    Attributes:
        results: Ranked list of matching documents, ordered by relevance.
    """

    msg: list[SearchResultItem]
