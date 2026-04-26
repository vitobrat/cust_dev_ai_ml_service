"""Base Qdrant repository with multi-tenant isolation via user_id payload."""

import uuid
from abc import ABC
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (  # noqa: WPS235
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    HnswConfigDiff,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from src.configs.consts import _USER_ID_PAYLOAD_KEY


class BaseQdrantRepository(ABC):
    """Abstract base repository for multi-tenant vector operations on a Qdrant collection.

    All documents live in a single shared collection. Multi-tenancy is achieved
    by storing the owner's user_id in each Point's payload and filtering by it
    on every read and delete operation.

    A KEYWORD payload index on user_id is created at startup so that filtered
    searches use the index instead of scanning the full payload.

    Attributes:
        _client: Shared async Qdrant client singleton.
        _collection_name: Target Qdrant collection name.
        _vector_size: Dimensionality of the stored embedding vectors.
        _hnsw_edge_size: Number of edges per HNSW graph node (m parameter).
        _hnsw_neighbour_size: Candidate pool size during HNSW construction (ef_construct).
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        vector_size: int,
        hnsw_edge_size: int,
        hnsw_neighbour_size: int,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._hnsw_edge_size = hnsw_edge_size
        self._hnsw_neighbour_size = hnsw_neighbour_size

    async def ensure_collection(self) -> None:
        """Create the collection and user_id payload index if they do not exist.

        Idempotent: safe to call on every application startup.
        The KEYWORD index on user_id enables O(log N) tenant-scoped filtering.
        """
        if await self._client.collection_exists(self._collection_name):
            await self._ensure_user_id_payload_index()
            return

        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=self._hnsw_edge_size, ef_construct=self._hnsw_neighbour_size),
        )
        await self._create_user_id_payload_index()

    async def is_collection_ready(self) -> bool:
        """Return whether the configured Qdrant collection is available."""
        return await self._client.collection_exists(self._collection_name)

    async def upsert(self, user_id: uuid.UUID, points: list[PointStruct]) -> None:
        """Insert or update points, stamping each with the owner's user_id.

        Existing payload fields on each point are preserved; user_id is stored
        as a string to match the KEYWORD index comparison type.

        Args:
            user_id: Owner identifier injected into every point's payload.
            points: Points to upsert. Existing points with the same id are overwritten.
        """
        for point in points:
            tmp_payload: dict[str, Any] = dict(point.payload) if point.payload else {}
            tmp_payload[_USER_ID_PAYLOAD_KEY] = str(user_id)
            point.payload = tmp_payload

        await self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    async def delete_by_ids(self, user_id: uuid.UUID, point_ids: list[uuid.UUID]) -> None:
        """Delete specific points owned by user_id.

        user_id is always included in the filter to prevent cross-user deletions.

        Args:
            user_id: Owner of the points to delete.
            point_ids: UUIDs of the points to remove.
        """
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key=_USER_ID_PAYLOAD_KEY,
                            match=MatchValue(value=str(user_id)),
                        ),
                        HasIdCondition(has_id=point_ids),
                    ],
                ),
            ),
        )

    async def delete_all_for_user(self, user_id: uuid.UUID) -> None:
        """Delete all points belonging to user_id.

        Args:
            user_id: Owner whose entire document set should be removed.
        """
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key=_USER_ID_PAYLOAD_KEY,
                            match=MatchValue(value=str(user_id)),
                        ),
                    ],
                ),
            ),
        )

    async def search(
        self,
        user_id: uuid.UUID,
        query_vector: list[float],
        limit: int = 10,
        extra_filter: Filter | None = None,
    ) -> list[ScoredPoint]:
        """Search for nearest vectors scoped to user_id.

        Args:
            user_id: Owner of the documents to search within.
            query_vector: Query embedding vector.
            limit: Maximum number of results to return.
            extra_filter: Optional additional Qdrant filter merged with user_id isolation.

        Returns:
            List of scored points ordered by relevance descending.
        """
        must_conditions: list[Any] = [
            FieldCondition(key=_USER_ID_PAYLOAD_KEY, match=MatchValue(value=str(user_id))),
        ]

        if extra_filter and extra_filter.must:
            must_conditions.extend(extra_filter.must)

        points_result = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            query_filter=Filter(must=must_conditions),
            limit=limit,
            with_payload=True,
        )
        return points_result.points

    async def _ensure_user_id_payload_index(self) -> None:
        """Create the user_id payload index when an existing collection lacks it."""
        collection_info = await self._client.get_collection(self._collection_name)
        payload_schema = collection_info.payload_schema or {}
        if _USER_ID_PAYLOAD_KEY in payload_schema:
            return

        await self._create_user_id_payload_index()

    async def _create_user_id_payload_index(self) -> None:
        """Create the KEYWORD payload index used for tenant filtering."""
        await self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name=_USER_ID_PAYLOAD_KEY,
            field_schema=PayloadSchemaType.KEYWORD,
        )
