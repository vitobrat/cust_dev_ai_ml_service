"""Fixtures for EmbeddingsService unit tests."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domains.embeddings.app.usecases.service import EmbeddingsService
from src.infrastructure.db.qdrant.embedding_repository import (
    EmbeddingRepository,
)
from src.infrastructure.triton.client import TritonClient


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
def mock_repository() -> MagicMock:
    """Return an EmbeddingRepository mock with all async methods pre-configured.

    Returns:
        MagicMock with upsert, delete_by_ids, and delete_all_for_user as AsyncMocks.
    """
    mock = MagicMock(spec=EmbeddingRepository)
    mock.upsert = AsyncMock()
    mock.delete_by_ids = AsyncMock()
    mock.delete_all_for_user = AsyncMock()
    return mock


@pytest.fixture
def embeddings_service(mock_triton: MagicMock, mock_repository: MagicMock) -> EmbeddingsService:
    """Return EmbeddingsService wired with mocked Triton and repository.

    Args:
        mock_triton: Mocked TritonClient.
        mock_repository: Mocked EmbeddingRepository.

    Returns:
        EmbeddingsService instance ready for unit testing.
    """
    return EmbeddingsService(
        repository=cast(EmbeddingRepository, mock_repository),
        triton_client=cast(TritonClient, mock_triton),
    )
