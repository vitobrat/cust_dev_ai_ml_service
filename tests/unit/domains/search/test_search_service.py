"""Unit tests for SearchService."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import ScoredPoint

from src.domains.search.app.usecases.service import SearchService
from src.domains.search.exceptions import SearchQueryError
from src.infrastructure.exceptions import EmbeddingError
from src.schemas.search import SearchResultItem


async def test_search_returns_list_of_search_result_items(
    search_service: SearchService,
    user_id: uuid.UUID,
) -> None:
    """search must return a list where every element is a SearchResultItem."""
    search_results = await search_service.search(user_id, query="test query", top_k=5)

    assert isinstance(search_results, list)
    assert all(isinstance(result_item, SearchResultItem) for result_item in search_results)


async def test_search_maps_scored_point_to_result_item_correctly(
    search_service: SearchService,
    mock_repository: MagicMock,
    mock_scored_point: ScoredPoint,
    user_id: uuid.UUID,
) -> None:
    """Each ScoredPoint field must be mapped 1-to-1 to the corresponding SearchResultItem field."""
    mock_repository.search = AsyncMock(return_value=[mock_scored_point])

    search_results = await search_service.search(user_id, query="query", top_k=1)

    assert len(search_results) == 1
    assert search_results[0].score == mock_scored_point.score
    assert search_results[0].payload == mock_scored_point.payload


async def test_search_result_count_matches_repository_output(
    search_service: SearchService,
    mock_repository: MagicMock,
    mock_scored_point: ScoredPoint,
    user_id: uuid.UUID,
) -> None:
    """The number of returned SearchResultItems must equal the number of ScoredPoints from Qdrant."""
    mock_repository.search = AsyncMock(return_value=[mock_scored_point, mock_scored_point])

    search_results = await search_service.search(user_id, query="query", top_k=10)

    assert len(search_results) == 2  # noqa: WPS432


async def test_search_calls_triton_with_query_wrapped_in_list(
    search_service: SearchService,
    mock_triton: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """Triton must receive the query as a single-element list, not a bare string."""
    query = "find relevant documents"

    await search_service.search(user_id, query=query, top_k=10)

    mock_triton.embed.assert_awaited_once_with([query])


async def test_search_passes_first_embedding_to_repository(
    search_service: SearchService,
    mock_repository: MagicMock,
    mock_triton: MagicMock,
    fake_vector: list[float],
    user_id: uuid.UUID,
) -> None:
    """Repository search must receive the first (and only) vector from Triton output."""
    mock_triton.embed = AsyncMock(return_value=[fake_vector])

    await search_service.search(user_id, query="query", top_k=5)

    query_vector: list[float] = mock_repository.search.call_args.args[1]
    assert query_vector == fake_vector


async def test_search_passes_top_k_as_limit_to_repository(
    search_service: SearchService,
    mock_repository: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """The top_k argument must be forwarded to repository.search as the limit keyword."""
    await search_service.search(user_id, query="query", top_k=7)

    limit: int = mock_repository.search.call_args.kwargs["limit"]
    assert limit == 7  # noqa: WPS432


async def test_search_triton_failure_raises_embedding_error(
    search_service: SearchService,
    mock_triton: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """Any Triton exception must be raised as EmbeddingError, not SearchQueryError.

    This tests the service's error boundary: Triton failures are infrastructure
    errors and are wrapped at the EmbeddingError level, not the search domain level.
    """
    mock_triton.embed = AsyncMock(side_effect=RuntimeError("gRPC connection lost"))

    with pytest.raises(EmbeddingError):
        await search_service.search(user_id, query="query", top_k=5)


async def test_search_empty_triton_response_raises_embedding_error(
    search_service: SearchService,
    mock_triton: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """An empty embedding list from Triton must raise EmbeddingError.

    An empty response means the model produced no output — this is treated as an
    infrastructure failure, not a domain-level search error.
    """
    mock_triton.embed = AsyncMock(return_value=[])

    with pytest.raises(EmbeddingError):
        await search_service.search(user_id, query="query", top_k=5)


async def test_search_repository_failure_raises_search_query_error(
    search_service: SearchService,
    mock_repository: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """Any exception from Qdrant during similarity search must be raised as SearchQueryError."""
    mock_repository.search = AsyncMock(side_effect=RuntimeError("Qdrant timeout"))

    with pytest.raises(SearchQueryError):
        await search_service.search(user_id, query="query", top_k=5)
