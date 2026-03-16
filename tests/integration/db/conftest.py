"""Fixtures for EmbeddingRepository integration tests.

Two repository fixtures serve different purposes:

- ``repository``       — raw repo, collection does NOT exist yet.
                         Use for tests that verify ensure_collection() behaviour.
- ``ready_repository`` — collection is already created via ensure_collection().
                         Use for tests that focus on data operations.

Both fixtures guarantee a clean collection state before and after each test
by deleting the collection in teardown (Qdrant has no SQL transactions).
"""

from collections.abc import AsyncGenerator

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
async def repository(qdrant_client: AsyncQdrantClient) -> AsyncGenerator[EmbeddingRepository, None]:
    """Return EmbeddingRepository without a pre-created collection.

    Setup: deletes the collection if it survived a previous failed test.
    Teardown: deletes the collection so the next test starts clean.

    Args:
        qdrant_client: Function-scoped AsyncQdrantClient from integration conftest.

    Yields:
        EmbeddingRepository instance connected to the test Qdrant container.
    """
    repo = EmbeddingRepository(
        client=qdrant_client,
        collection_name=_TEST_COLLECTION,
        vector_size=_TEST_VECTOR_SIZE,
        hnsw_edge_size=_TEST_HNSW_M,
        hnsw_neighbour_size=_TEST_HNSW_EF,
    )
    if await qdrant_client.collection_exists(_TEST_COLLECTION):
        await qdrant_client.delete_collection(_TEST_COLLECTION)

    yield repo

    if await qdrant_client.collection_exists(_TEST_COLLECTION):
        await qdrant_client.delete_collection(_TEST_COLLECTION)


@pytest.fixture
async def ready_repository(repository: EmbeddingRepository) -> EmbeddingRepository:
    """Return EmbeddingRepository with the collection already created.

    Args:
        repository: Base repository fixture with guaranteed clean state.

    Returns:
        EmbeddingRepository with an empty collection ready for data operations.
    """
    await repository.ensure_collection()
    return repository
