"""Search application service."""

import time
import uuid

from src.configs.log.logger import get_logger
from src.domains.search.exceptions import SearchQueryError
from src.infrastructure.db.qdrant.embedding_repository import (
    EmbeddingRepository,
)
from src.infrastructure.exceptions import EmbeddingError
from src.infrastructure.triton.client import TritonClient
from src.schemas.search import SearchResultItem

_QUERY_PREFIX = "query: "


class SearchService:
    """Orchestrates query embedding and Qdrant similarity search.

    Attributes:
        _repository: Repository for similarity search operations.
        _triton_client: Client for Triton embedding model inference.
    """

    def __init__(self, repository: EmbeddingRepository, triton_client: TritonClient) -> None:
        """Initialize the service with injected dependencies.

        Args:
            repository: Qdrant repository for the embeddings collection.
            triton_client: Triton gRPC client for embedding model inference.
        """
        self._repository = repository
        self._triton_client = triton_client
        self._logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    async def search(self, user_id: uuid.UUID, query: str, top_k: int) -> list[SearchResultItem]:
        """Embed a query and retrieve the most similar documents from Qdrant.

        Args:
            user_id: Owner of the document space to search within.
            query: Natural language query text.
            top_k: Maximum number of results to return.

        Returns:
            Ranked list of matching search results ordered by relevance.

        Raises:
            SearchQueryError: If embedding generation or Qdrant search fails.
        """
        started_at = time.perf_counter()
        query_for_embedding = self._as_query(query)

        try:
            vectors = await self._triton_client.embed([query_for_embedding])
        except Exception as embed_exc:
            raise EmbeddingError(str(embed_exc)) from embed_exc

        if len(vectors) > 0:
            query_vector = vectors[0]
        else:
            raise EmbeddingError(f"Empty or not valid output vector from Triton: {vectors}")

        try:
            scored_points = await self._repository.search(user_id, query_vector, limit=top_k)
        except Exception as search_exc:
            raise SearchQueryError(str(search_exc)) from search_exc

        search_results = [
            SearchResultItem(
                id=uuid.UUID(str(point.id)),
                score=point.score,
                payload=point.payload or {},
            )
            for point in scored_points
        ]

        elapsed = time.perf_counter() - started_at
        self._logger.info("Semantic search completed in %.4fs with %s results", elapsed, len(search_results))

        return search_results

    @staticmethod
    def _as_query(query: str) -> str:
        """Apply the E5 query prefix expected for search queries."""
        if query.startswith(_QUERY_PREFIX):
            return query
        return f"{_QUERY_PREFIX}{query}"
