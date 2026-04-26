"""Unit tests for FastAPI readiness helpers."""

from unittest.mock import AsyncMock, MagicMock

from src.infrastructure.health import build_readiness_payload


async def test_build_readiness_payload_reports_ready_when_dependencies_are_ready() -> None:
    """Readiness payload must be healthy only when Qdrant and Triton are ready."""
    repository = MagicMock()
    repository.is_collection_ready = AsyncMock(return_value=True)
    triton_client = MagicMock()
    triton_client.is_ready = AsyncMock(return_value=True)

    payload = await build_readiness_payload(repository, triton_client)

    assert payload == {"qdrant": True, "triton": True, "ready": True}


async def test_build_readiness_payload_reports_not_ready_when_dependency_fails() -> None:
    """Readiness payload must expose the failing dependency."""
    repository = MagicMock()
    repository.is_collection_ready = AsyncMock(return_value=True)
    triton_client = MagicMock()
    triton_client.is_ready = AsyncMock(side_effect=RuntimeError("triton down"))

    payload = await build_readiness_payload(repository, triton_client)

    assert payload == {"qdrant": True, "triton": False, "ready": False}
