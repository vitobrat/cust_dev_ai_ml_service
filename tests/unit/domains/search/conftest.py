"""Fixtures for SearchService unit tests."""

import uuid
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import ScoredPoint

from src.domains.search.app.usecases.service import SearchService
from src.infrastructure.db.qdrant.embedding_repository import (
    EmbeddingRepository,
)
from src.infrastructure.triton.client import TritonClient

_STUB_SCORE: float = 0.95  # noqa: WPS432


@pytest.fixture
def mock_triton(fake_vector: list[float]) -> MagicMock:
    """Return a TritonClient mock whose embed() returns a single fake vector.

    Args:
        fake_vector: Shared deterministic embedding vector from root conftest.

    Returns:
        MagicMock configured to act as a TritonClient.
    """
    mock = MagicMock(spec=TritonClient)
    mock.embed = AsyncMock(return_value=[fake_vector])
    return mock


@pytest.fixture
def mock_scored_point(user_id: uuid.UUID) -> ScoredPoint:
    """Return a ScoredPoint stub representing a single Qdrant search result.

    Args:
        user_id: Owner UUID used as the user_id payload field value.

    Returns:
        A ScoredPoint with a deterministic score and a non-empty payload.
    """
    return ScoredPoint(
        id=str(uuid.uuid4()),
        version=0,
        score=_STUB_SCORE,
        payload={"text": "relevant document", "user_id": str(user_id)},
    )


@pytest.fixture
def mock_repository(mock_scored_point: ScoredPoint) -> MagicMock:
    """Return an EmbeddingRepository mock whose search() returns one ScoredPoint.

    Args:
        mock_scored_point: Pre-built ScoredPoint used as the default search result.

    Returns:
        MagicMock configured to act as an EmbeddingRepository.
    """
    mock = MagicMock(spec=EmbeddingRepository)
    mock.search = AsyncMock(return_value=[mock_scored_point])
    return mock


@pytest.fixture
def search_service(mock_triton: MagicMock, mock_repository: MagicMock) -> SearchService:
    """Return SearchService wired with mocked Triton and repository.

    Args:
        mock_triton: Mocked TritonClient.
        mock_repository: Mocked EmbeddingRepository.

    Returns:
        SearchService instance ready for unit testing.
    """
    return SearchService(
        repository=cast(EmbeddingRepository, mock_repository),
        triton_client=cast(TritonClient, mock_triton),
    )
