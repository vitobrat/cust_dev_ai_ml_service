"""Unit tests for BaseQdrantRepository via EmbeddingRepository.

These tests verify the multi-tenant isolation logic, filter construction,
and collection management behaviour implemented in BaseQdrantRepository.
The concrete EmbeddingRepository subclass is used as the test target
because BaseQdrantRepository is abstract.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

from qdrant_client.models import (  # noqa: WPS235
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    MatchValue,
    PointStruct,
)

from src.configs.consts import _USER_ID_PAYLOAD_KEY
from src.infrastructure.db.qdrant.embedding_repository import (
    EmbeddingRepository,
)

_TEST_SEARCH_LIMIT: int = 15  # noqa: WPS432


async def test_ensure_collection_creates_collection_when_absent(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
) -> None:
    """ensure_collection must call create_collection when the collection does not exist."""
    mock_qdrant_client.collection_exists = AsyncMock(return_value=False)

    await repository.ensure_collection()

    mock_qdrant_client.create_collection.assert_awaited_once()


async def test_ensure_collection_skips_creation_when_already_present(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
) -> None:
    """ensure_collection must not call create_collection when the collection already exists."""
    mock_qdrant_client.collection_exists = AsyncMock(return_value=True)

    await repository.ensure_collection()

    mock_qdrant_client.create_collection.assert_not_awaited()


async def test_ensure_collection_creates_user_id_index_when_collection_exists_without_it(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
) -> None:
    """ensure_collection must repair a missing user_id index on an existing collection."""
    collection_info = MagicMock()
    collection_info.payload_schema = {}
    mock_qdrant_client.collection_exists = AsyncMock(return_value=True)
    mock_qdrant_client.get_collection = AsyncMock(return_value=collection_info)

    await repository.ensure_collection()

    mock_qdrant_client.create_collection.assert_not_awaited()
    field_name: str = mock_qdrant_client.create_payload_index.call_args.kwargs["field_name"]
    assert field_name == _USER_ID_PAYLOAD_KEY


async def test_ensure_collection_creates_keyword_index_on_user_id_field(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
) -> None:
    """ensure_collection must create a payload index on the user_id field for fast filtering."""
    mock_qdrant_client.collection_exists = AsyncMock(return_value=False)

    await repository.ensure_collection()

    field_name: str = mock_qdrant_client.create_payload_index.call_args.kwargs["field_name"]
    assert field_name == _USER_ID_PAYLOAD_KEY


async def test_upsert_stamps_user_id_into_every_point_payload(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
    user_id: uuid.UUID,
    fake_vector: list[float],
) -> None:
    """upsert must inject user_id as a string into each PointStruct payload in-place."""
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=fake_vector, payload={"text": "doc A"}),
        PointStruct(id=str(uuid.uuid4()), vector=fake_vector, payload={"text": "doc B"}),
    ]

    await repository.upsert(user_id, points)

    assert all(point.payload.get(_USER_ID_PAYLOAD_KEY) == str(user_id) for point in points)


async def test_upsert_preserves_existing_payload_fields_alongside_user_id(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
    user_id: uuid.UUID,
    fake_vector: list[float],
) -> None:
    """upsert must not overwrite pre-existing payload fields; only user_id is added."""
    original_text = "original document content"
    point_id = str(uuid.uuid4())
    point = PointStruct(id=point_id, vector=fake_vector, payload={"text": original_text})

    await repository.upsert(user_id, [point])

    assert point.payload.get("text") == original_text
    assert point.payload.get(_USER_ID_PAYLOAD_KEY) == str(user_id)


async def test_upsert_handles_point_with_empty_payload(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
    user_id: uuid.UUID,
    fake_vector: list[float],
) -> None:
    """upsert must correctly stamp user_id even when the point has an empty payload."""
    point = PointStruct(id=str(uuid.uuid4()), vector=fake_vector, payload={})

    await repository.upsert(user_id, [point])

    assert point.payload.get(_USER_ID_PAYLOAD_KEY) == str(user_id)


async def test_delete_by_ids_filter_contains_user_id_field_condition(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """delete_by_ids must include a FieldCondition on user_id to prevent cross-tenant deletions."""
    point_id = uuid.uuid4()

    await repository.delete_by_ids(user_id, [point_id])

    selector: FilterSelector = mock_qdrant_client.delete.call_args.kwargs["points_selector"]
    must = selector.filter.must or []
    field_conditions = [cond for cond in must if isinstance(cond, FieldCondition)]
    expected_match = MatchValue(value=str(user_id))
    assert any(cond.key == _USER_ID_PAYLOAD_KEY and cond.match == expected_match for cond in field_conditions)


async def test_delete_by_ids_filter_contains_has_id_condition_with_requested_ids(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """delete_by_ids filter must include a HasIdCondition listing the exact point IDs to delete."""
    point_id = uuid.uuid4()

    await repository.delete_by_ids(user_id, [point_id])

    selector: FilterSelector = mock_qdrant_client.delete.call_args.kwargs["points_selector"]
    must = selector.filter.must or []
    has_id_conditions = [cond for cond in must if isinstance(cond, HasIdCondition)]

    assert len(has_id_conditions) == 1
    assert point_id in has_id_conditions[0].has_id


async def test_delete_all_for_user_filter_contains_only_user_id_condition(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
    user_id: uuid.UUID,
) -> None:
    """delete_all_for_user filter must contain exactly one condition: the user_id FieldCondition.

    Having more conditions could accidentally narrow the deletion scope and leave
    orphaned points in the collection.
    """
    await repository.delete_all_for_user(user_id)

    selector: FilterSelector = mock_qdrant_client.delete.call_args.kwargs["points_selector"]
    must = selector.filter.must or []

    assert len(must) == 1
    assert isinstance(must[0], FieldCondition)
    assert must[0].key == _USER_ID_PAYLOAD_KEY


async def test_search_query_filter_includes_user_id_field_condition(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
    user_id: uuid.UUID,
    fake_vector: list[float],
) -> None:
    """search must scope results to the owner by including a user_id FieldCondition."""
    await repository.search(user_id, query_vector=fake_vector, limit=5)

    query_filter: Filter = mock_qdrant_client.query_points.call_args.kwargs["query_filter"]
    must = query_filter.must or []
    field_conditions = [cond for cond in must if isinstance(cond, FieldCondition)]
    expected_match = MatchValue(value=str(user_id))
    assert any(cond.key == _USER_ID_PAYLOAD_KEY and cond.match == expected_match for cond in field_conditions)


async def test_search_merges_extra_filter_must_conditions_with_user_id(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
    user_id: uuid.UUID,
    fake_vector: list[float],
) -> None:
    """extra_filter.must conditions must be appended to the user_id condition, not replace it."""
    extra = Filter(must=[FieldCondition(key="category", match=MatchValue(value="tech"))])

    await repository.search(user_id, query_vector=fake_vector, limit=5, extra_filter=extra)

    query_filter: Filter = mock_qdrant_client.query_points.call_args.kwargs["query_filter"]
    must = query_filter.must or []
    condition_keys = [cond.key for cond in must if isinstance(cond, FieldCondition)]

    assert _USER_ID_PAYLOAD_KEY in condition_keys
    assert "category" in condition_keys


async def test_search_forwards_limit_argument_to_qdrant_client(
    repository: EmbeddingRepository,
    mock_qdrant_client: MagicMock,
    user_id: uuid.UUID,
    fake_vector: list[float],
) -> None:
    """search must pass the limit argument through to the underlying Qdrant client."""
    await repository.search(user_id, query_vector=fake_vector, limit=_TEST_SEARCH_LIMIT)

    limit: int = mock_qdrant_client.query_points.call_args.kwargs["limit"]
    assert limit == _TEST_SEARCH_LIMIT
