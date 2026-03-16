"""Unit tests for EmbeddingsService."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import PointStruct

from src.domains.embeddings.app.usecases.service import EmbeddingsService
from src.domains.embeddings.exceptions import (
    EmbeddingDeleteError,
    EmbeddingUpsertError,
)
from src.infrastructure.exceptions import EmbeddingError


async def test_upsert_returns_count_equal_to_texts_length(
    embeddings_service: EmbeddingsService,
    mock_triton: MagicMock,
    user_id: uuid.UUID,
    sample_texts: list[str],
    fake_vector: list[float],
) -> None:
    """upsert must return the number of documents processed, one per input text."""
    mock_triton.embed = AsyncMock(return_value=[fake_vector for _ in sample_texts])

    count = await embeddings_service.upsert(user_id, sample_texts)

    assert count == len(sample_texts)


async def test_upsert_calls_triton_with_all_texts(
    embeddings_service: EmbeddingsService,
    mock_triton: MagicMock,
    user_id: uuid.UUID,
    sample_texts: list[str],
) -> None:
    """Triton must be called exactly once with the complete list of input texts."""
    await embeddings_service.upsert(user_id, sample_texts)

    mock_triton.embed.assert_awaited_once_with(sample_texts)


async def test_upsert_calls_repository_with_correct_user_id(
    embeddings_service: EmbeddingsService,
    mock_repository: MagicMock,
    mock_triton: MagicMock,
    user_id: uuid.UUID,
    sample_texts: list[str],
    fake_vector: list[float],
) -> None:
    """Repository upsert must receive the owner's user_id as its first argument."""
    mock_triton.embed = AsyncMock(return_value=[fake_vector for _ in sample_texts])

    await embeddings_service.upsert(user_id, sample_texts)

    actual_user_id: uuid.UUID = mock_repository.upsert.call_args.args[0]
    assert actual_user_id == user_id


async def test_upsert_creates_one_point_per_text(
    embeddings_service: EmbeddingsService,
    mock_repository: MagicMock,
    mock_triton: MagicMock,
    user_id: uuid.UUID,
    sample_texts: list[str],
    fake_vector: list[float],
) -> None:
    """Each input text must produce exactly one PointStruct passed to the repository."""
    mock_triton.embed = AsyncMock(return_value=[fake_vector for _ in sample_texts])

    await embeddings_service.upsert(user_id, sample_texts)

    points: list[PointStruct] = mock_repository.upsert.call_args.args[1]
    assert len(points) == len(sample_texts)


async def test_upsert_stores_original_text_in_point_payload(
    embeddings_service: EmbeddingsService,
    mock_repository: MagicMock,
    mock_triton: MagicMock,
    user_id: uuid.UUID,
    sample_texts: list[str],
    fake_vector: list[float],
) -> None:
    """Each PointStruct payload must contain the source text under the 'text' key."""
    mock_triton.embed = AsyncMock(return_value=[fake_vector for _ in sample_texts])

    await embeddings_service.upsert(user_id, sample_texts)

    points: list[PointStruct] = mock_repository.upsert.call_args.args[1]
    stored_texts = [point.payload["text"] for point in points]
    assert stored_texts == sample_texts


async def test_upsert_triton_failure_raises_embedding_error(
    embeddings_service: EmbeddingsService,
    mock_triton: MagicMock,
    user_id: uuid.UUID,
    sample_texts: list[str],
) -> None:
    """Any exception from Triton during embedding must be raised as EmbeddingError.

    Note: the service wraps this in EmbeddingError (base), not EmbeddingUpsertError,
    because the failure occurs before the repository is involved.
    """
    mock_triton.embed = AsyncMock(side_effect=RuntimeError("gRPC timeout"))

    with pytest.raises(EmbeddingError):
        await embeddings_service.upsert(user_id, sample_texts)


async def test_upsert_repository_failure_raises_upsert_error(
    embeddings_service: EmbeddingsService,
    mock_repository: MagicMock,
    user_id: uuid.UUID,
    sample_texts: list[str],
) -> None:
    """Any exception from the repository during upsert must be raised as EmbeddingUpsertError."""
    mock_repository.upsert = AsyncMock(side_effect=RuntimeError("Qdrant unavailable"))

    with pytest.raises(EmbeddingUpsertError):
        await embeddings_service.upsert(user_id, sample_texts)


async def test_delete_by_id_delegates_to_repository_with_single_item_list(
    embeddings_service: EmbeddingsService,
    mock_repository: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """delete_by_id must call repository.delete_by_ids with [point_id] and the owner user_id."""
    point_id = uuid.uuid4()

    await embeddings_service.delete_by_id(user_id, point_id)

    mock_repository.delete_by_ids.assert_awaited_once_with(user_id, [point_id])


async def test_delete_by_id_repository_failure_raises_delete_error(
    embeddings_service: EmbeddingsService,
    mock_repository: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """Any exception from the repository during delete_by_id must be raised as EmbeddingDeleteError."""
    mock_repository.delete_by_ids = AsyncMock(side_effect=RuntimeError("Qdrant error"))

    with pytest.raises(EmbeddingDeleteError):
        await embeddings_service.delete_by_id(user_id, uuid.uuid4())


async def test_delete_all_delegates_to_repository_with_user_id(
    embeddings_service: EmbeddingsService,
    mock_repository: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """delete_all must call repository.delete_all_for_user with the correct user_id."""
    await embeddings_service.delete_all(user_id)

    mock_repository.delete_all_for_user.assert_awaited_once_with(user_id)


async def test_delete_all_repository_failure_raises_delete_error(
    embeddings_service: EmbeddingsService,
    mock_repository: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """Any exception from the repository during delete_all must be raised as EmbeddingDeleteError."""
    mock_repository.delete_all_for_user = AsyncMock(side_effect=RuntimeError("Qdrant error"))

    with pytest.raises(EmbeddingDeleteError):
        await embeddings_service.delete_all(user_id)
