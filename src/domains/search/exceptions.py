"""Search domain exception hierarchy."""


class SearchError(Exception):
    """Base exception for all search domain errors."""


class SearchQueryError(SearchError):
    """Raised when query Qdrant similarity search fails."""
