"""Fixtures for infrastructure-layer unit tests."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import AsyncQdrantClient

from src.infrastructure.db.qdrant.embedding_repository import (
    EmbeddingRepository,
)

_TEST_COLLECTION: str = "test_documents"
_TEST_VECTOR_SIZE: int = 5
_TEST_HNSW_M: int = 4
_TEST_HNSW_EF: int = 16


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Return an AsyncQdrantClient mock with all async operations pre-configured.

    Each method is an AsyncMock so it can be awaited inside repository methods.
    collection_exists defaults to False so ensure_collection triggers creation.

    Returns:
        MagicMock configured to act as an AsyncQdrantClient.
    """
    mock = MagicMock(spec=AsyncQdrantClient)
    mock.collection_exists = AsyncMock(return_value=False)
    mock.create_collection = AsyncMock()
    mock.create_payload_index = AsyncMock()
    mock.upsert = AsyncMock()
    mock.delete = AsyncMock()
    mock_result = MagicMock()
    mock_result.points = []
    mock.query_points = AsyncMock(return_value=mock_result)
    return mock


@pytest.fixture
def repository(mock_qdrant_client: MagicMock) -> EmbeddingRepository:
    """Return EmbeddingRepository wired with a mocked Qdrant client.

    Args:
        mock_qdrant_client: Mocked AsyncQdrantClient.

    Returns:
        EmbeddingRepository ready for isolated unit testing.
    """
    return EmbeddingRepository(
        client=cast(AsyncQdrantClient, mock_qdrant_client),
        collection_name=_TEST_COLLECTION,
        vector_size=_TEST_VECTOR_SIZE,
        hnsw_edge_size=_TEST_HNSW_M,
        hnsw_neighbour_size=_TEST_HNSW_EF,
    )
