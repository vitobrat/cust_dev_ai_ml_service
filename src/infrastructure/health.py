"""Import-safe readiness helpers for external service checks."""

from typing import TypedDict

from src.configs.log.logger import get_logger
from src.infrastructure.db.qdrant.embedding_repository import (
    EmbeddingRepository,
)
from src.infrastructure.triton.client import TritonClient

_logger = get_logger(__name__)


class ReadinessPayload(TypedDict):
    """Dependency readiness flags returned by the readiness endpoint."""

    qdrant: bool
    triton: bool
    ready: bool


async def build_readiness_payload(
    repository: EmbeddingRepository,
    triton_client: TritonClient,
) -> ReadinessPayload:
    """Check Qdrant collection availability and Triton model readiness."""
    try:
        qdrant_ready = await repository.is_collection_ready()
    except Exception as exc:
        _logger.warning("Qdrant readiness check failed: %s", exc)
        qdrant_ready = False

    try:
        triton_ready = await triton_client.is_ready()
    except Exception as exc:
        _logger.warning("Triton readiness check failed: %s", exc)
        triton_ready = False

    return {
        "qdrant": qdrant_ready,
        "triton": triton_ready,
        "ready": qdrant_ready and triton_ready,
    }
