"""Concrete Qdrant repository for user document embeddings."""

from qdrant_client import AsyncQdrantClient

from src.infrastructure.db.qdrant.repository import BaseQdrantRepository


class EmbeddingRepository(BaseQdrantRepository):
    """Qdrant repository for multi-tenant document embedding storage.

    Inherits all vector operations (upsert, search, delete) from
    BaseQdrantRepository. HNSW index parameters are fixed to sensible
    defaults for general-purpose embedding similarity search.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        vector_size: int,
        hnsw_edge_size: int,
        hnsw_neighbour_size: int,
    ) -> None:
        """Initialize with a shared Qdrant client and collection settings.

        Args:
            client: Shared async Qdrant client singleton.
            collection_name: Target collection for document embeddings.
            vector_size: Dimensionality of stored embedding vectors.
            hnsw_edge_size: Number of bi-directional links per HNSW graph node (m parameter).
            hnsw_neighbour_size: Candidate pool size during HNSW construction (ef_construct).
        """
        super().__init__(
            client=client,
            collection_name=collection_name,
            vector_size=vector_size,
            hnsw_edge_size=hnsw_edge_size,
            hnsw_neighbour_size=hnsw_neighbour_size,
        )
