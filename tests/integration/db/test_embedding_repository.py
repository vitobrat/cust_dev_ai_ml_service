"""Integration tests for EmbeddingRepository against a real Qdrant instance.

These tests verify behaviour that cannot be confirmed with mocks:
- Real collection creation and KEYWORD payload index setup.
- Actual vector upsert persistence and retrieval via HNSW search.
- Correct cosine similarity ordering over real indexed data.
- Multi-tenant isolation: user_A queries must never return user_B points.
- Correct scoping of delete operations to a single tenant.

All tests use 5-dimensional unit vectors so the search ordering is
mathematically unambiguous and independent of HNSW approximation quality.
"""

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from src.infrastructure.db.qdrant.embedding_repository import (
    EmbeddingRepository,
)

# Deterministic unit vectors that give unambiguous cosine similarity results.
# sim([1,0,0,0,0], [1,0,0,0,0]) = 1.0  (identical)
# sim([1,0,0,0,0], [0,0,0,0,1]) ≈ 0.0  (orthogonal)
_VEC_A: list[float] = [1.0, 0.0, 0.0, 0.0, 0.0]  # noqa: WPS407, WPS358
_VEC_B: list[float] = [0.0, 0.0, 0.0, 0.0, 1.0]  # noqa: WPS407, WPS358
_VEC_C: list[float] = [0.0, 1.0, 0.0, 0.0, 0.0]  # noqa: WPS407, WPS358


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------


async def test_ensure_collection_creates_collection_in_qdrant(
    repository: EmbeddingRepository,
    qdrant_client: AsyncQdrantClient,
) -> None:
    """ensure_collection must create the collection so it is discoverable by the Qdrant client."""
    assert not await qdrant_client.collection_exists("test_documents")

    await repository.ensure_collection()

    assert await qdrant_client.collection_exists("test_documents")


async def test_ensure_collection_is_idempotent(
    repository: EmbeddingRepository,
) -> None:
    """Calling ensure_collection twice must not raise an exception."""
    await repository.ensure_collection()
    await repository.ensure_collection()  # must not raise


# ---------------------------------------------------------------------------
# Upsert and basic retrieval
# ---------------------------------------------------------------------------


async def test_upsert_persists_points_that_are_retrievable_via_search(
    ready_repository: EmbeddingRepository,
    user_id: uuid.UUID,
) -> None:
    """Points upserted to Qdrant must be returned by a matching similarity search."""
    point_id = str(uuid.uuid4())
    points = [PointStruct(id=point_id, vector=_VEC_A, payload={"text": "alpha doc"})]

    await ready_repository.upsert(user_id, points)
    found_points = await ready_repository.search(user_id, query_vector=_VEC_A, limit=10)

    returned_ids = [str(hit.id) for hit in found_points]
    assert point_id in returned_ids


async def test_upsert_stamps_user_id_in_persisted_payload(
    ready_repository: EmbeddingRepository,
    user_id: uuid.UUID,
) -> None:
    """After upsert, every stored point payload must contain the owner's user_id as a string."""
    stub_point_id = str(uuid.uuid4())
    points = [PointStruct(id=stub_point_id, vector=_VEC_A, payload={"text": "doc"})]

    await ready_repository.upsert(user_id, points)
    found_points = await ready_repository.search(user_id, query_vector=_VEC_A, limit=1)

    assert len(found_points) == 1
    assert found_points[0].payload.get("user_id") == str(user_id)


# ---------------------------------------------------------------------------
# Search behaviour
# ---------------------------------------------------------------------------


async def test_search_returns_results_ordered_by_descending_score(
    ready_repository: EmbeddingRepository,
    user_id: uuid.UUID,
) -> None:
    """Results must be ordered from most to least similar to the query vector."""
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=_VEC_A, payload={"text": "relevant"}),
        PointStruct(id=str(uuid.uuid4()), vector=_VEC_B, payload={"text": "unrelated"}),
    ]
    await ready_repository.upsert(user_id, points)

    # Query identical to _VEC_A — its point must score highest
    found_points = await ready_repository.search(user_id, query_vector=_VEC_A, limit=2)

    assert len(found_points) == 2
    assert found_points[0].score >= found_points[1].score


async def test_search_respects_top_k_limit(
    ready_repository: EmbeddingRepository,
    user_id: uuid.UUID,
) -> None:
    """search must return at most top_k results even when more points exist."""
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=_VEC_A, payload={}),
        PointStruct(id=str(uuid.uuid4()), vector=_VEC_B, payload={}),
        PointStruct(id=str(uuid.uuid4()), vector=_VEC_C, payload={}),
    ]
    await ready_repository.upsert(user_id, points)

    found_points = await ready_repository.search(user_id, query_vector=_VEC_A, limit=2)

    assert len(found_points) <= 2  # noqa: WPS432


async def test_search_is_scoped_to_requesting_user(
    ready_repository: EmbeddingRepository,
    user_id: uuid.UUID,
    other_user_id: uuid.UUID,
) -> None:
    """Search results for user_A must never contain points owned by user_B.

    This is the core multi-tenancy invariant. A failure here means data leaks
    between tenants, which is a critical security issue.
    """
    user_a_point_id = str(uuid.uuid4())
    user_b_point_id = str(uuid.uuid4())

    await ready_repository.upsert(
        user_id,
        [PointStruct(id=user_a_point_id, vector=_VEC_A, payload={"text": "user A doc"})],
    )
    await ready_repository.upsert(
        other_user_id,
        [PointStruct(id=user_b_point_id, vector=_VEC_A, payload={"text": "user B doc"})],
    )

    found_points = await ready_repository.search(user_id, query_vector=_VEC_A, limit=10)
    returned_ids = [str(hit.id) for hit in found_points]

    assert user_a_point_id in returned_ids
    assert user_b_point_id not in returned_ids


# ---------------------------------------------------------------------------
# Delete behaviour
# ---------------------------------------------------------------------------


async def test_delete_by_ids_removes_only_the_specified_point(
    ready_repository: EmbeddingRepository,
    user_id: uuid.UUID,
) -> None:
    """delete_by_ids must remove the targeted point while leaving other points intact."""
    point_to_delete = str(uuid.uuid4())
    point_to_keep = str(uuid.uuid4())
    points = [
        PointStruct(id=point_to_delete, vector=_VEC_A, payload={}),
        PointStruct(id=point_to_keep, vector=_VEC_B, payload={}),
    ]
    await ready_repository.upsert(user_id, points)

    await ready_repository.delete_by_ids(user_id, [uuid.UUID(point_to_delete)])

    found_points = await ready_repository.search(user_id, query_vector=_VEC_A, limit=10)
    returned_ids = [str(hit.id) for hit in found_points]

    assert point_to_delete not in returned_ids
    assert point_to_keep in returned_ids


async def test_delete_all_removes_only_the_target_user_points(
    ready_repository: EmbeddingRepository,
    user_id: uuid.UUID,
    other_user_id: uuid.UUID,
) -> None:
    """delete_all_for_user must not touch points owned by a different user.

    Validates that the user_id filter in the delete selector correctly
    scopes the deletion to a single tenant.
    """
    user_b_point_id = str(uuid.uuid4())

    await ready_repository.upsert(
        user_id,
        [PointStruct(id=str(uuid.uuid4()), vector=_VEC_A, payload={})],
    )
    await ready_repository.upsert(
        other_user_id,
        [PointStruct(id=user_b_point_id, vector=_VEC_A, payload={})],
    )

    await ready_repository.delete_all_for_user(user_id)

    # user_A data is gone
    user_a_results = await ready_repository.search(user_id, query_vector=_VEC_A, limit=10)
    assert len(user_a_results) == 0

    # user_B data is untouched
    user_b_results = await ready_repository.search(other_user_id, query_vector=_VEC_A, limit=10)
    returned_ids = [str(hit.id) for hit in user_b_results]
    assert user_b_point_id in returned_ids
