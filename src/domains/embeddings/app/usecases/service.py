"""Embeddings application service."""

import time
import uuid

from qdrant_client.models import PointStruct

from src.configs.log.logger import get_logger
from src.domains.embeddings.exceptions import (
    EmbeddingDeleteError,
    EmbeddingUpsertError,
)
from src.infrastructure.db.qdrant.embedding_repository import (
    EmbeddingRepository,
)
from src.infrastructure.exceptions import EmbeddingError
from src.infrastructure.triton.client import TritonClient

_PASSAGE_PREFIX = "passage: "


class EmbeddingsService:
    """Orchestrates text embedding generation and Qdrant upsert operations.

    Attributes:
        _repository: Repository for vector storage operations.
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

    async def upsert(self, user_id: uuid.UUID, texts: list[str]) -> int:
        """Embed texts via Triton and upsert them into Qdrant for the given user.

        Args:
            user_id: Owner of the documents being indexed.
            texts: Raw text documents to embed and store.

        Returns:
            Number of documents upserted.

        Raises:
            EmbeddingError: If Triton inference fails.
            EmbeddingUpsertError: If Qdrant upsert fails.
        """
        started_at = time.perf_counter()
        texts_for_embedding = [self._as_passage(text) for text in texts]

        try:
            embeddings = await self._triton_client.embed(texts_for_embedding)
        except Exception as embedding_exception:
            raise EmbeddingError(str(embedding_exception)) from embedding_exception

        points = [
            PointStruct(id=uuid.uuid4(), vector=embedding, payload={"text": text})
            for text, embedding in zip(texts, embeddings)
        ]

        try:
            await self._repository.upsert(user_id, points)
        except Exception as upsert_exception:
            raise EmbeddingUpsertError(str(upsert_exception)) from upsert_exception

        elapsed = time.perf_counter() - started_at
        self._logger.info("Embedding upsert completed in %.4fs for %s texts", elapsed, len(points))

        return len(points)

    async def delete_by_id(self, user_id: uuid.UUID, point_id: uuid.UUID) -> None:
        """Delete a single embedding point owned by the given user.

        Args:
            user_id: Owner of the point to delete.
            point_id: Qdrant point identifier to remove.

        Raises:
            EmbeddingDeleteError: If Qdrant deletion fails.
        """
        try:
            await self._repository.delete_by_ids(user_id, [point_id])
        except Exception as delete_exception:
            raise EmbeddingDeleteError(str(delete_exception)) from delete_exception

    async def delete_all(self, user_id: uuid.UUID) -> None:
        """Delete all embeddings belonging to the given user.

        Args:
            user_id: Owner whose entire document set should be removed.

        Raises:
            EmbeddingDeleteError: If Qdrant deletion fails.
        """
        try:
            await self._repository.delete_all_for_user(user_id)
        except Exception as delete_exception:
            raise EmbeddingDeleteError(str(delete_exception)) from delete_exception

    @staticmethod
    def _as_passage(text: str) -> str:
        """Apply the E5 passage prefix expected for indexed documents."""
        if text.startswith(_PASSAGE_PREFIX):
            return text
        return f"{_PASSAGE_PREFIX}{text}"
