"""Unit tests for EmbeddingsWorkerHandler."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

from src.domains.embeddings.app.workers.handler import EmbeddingsWorkerHandler
from src.infrastructure.exceptions import EmbeddingError
from src.schemas.api_base import StatusType
from src.schemas.embeddings import EmbeddingAction

_REPLY_TO = "reply-queue"
_CORRELATION_ID = "corr-123"


def _last_reply_payload(rabbitmq: MagicMock) -> dict[str, object]:
    """Return the last payload passed to publish_reply()."""
    return rabbitmq.publish_reply.call_args.kwargs["payload"]


async def test_upsert_success_reply_is_json_serializable(user_id: uuid.UUID) -> None:
    """Successful upsert replies must contain JSON primitives, not Enum objects."""
    service = MagicMock()
    service.upsert = AsyncMock(return_value=2)
    rabbitmq = MagicMock()
    rabbitmq.publish_reply = AsyncMock()
    worker_handler = EmbeddingsWorkerHandler(service=service, rabbitmq=rabbitmq)

    await worker_handler.embeddings_handle(
        _REPLY_TO,
        _CORRELATION_ID,
        {
            "action": EmbeddingAction.UPSERT,
            "user_id": str(user_id),
            "texts": ["first", "second"],
        },
    )

    payload = _last_reply_payload(rabbitmq)
    json.dumps(payload)
    assert payload["status"] == StatusType.SUCCESS.value


async def test_unknown_action_error_reply_is_json_serializable(user_id: uuid.UUID) -> None:
    """Unknown action errors must still be returned as JSON-serializable replies."""
    service = MagicMock()
    rabbitmq = MagicMock()
    rabbitmq.publish_reply = AsyncMock()
    worker_handler = EmbeddingsWorkerHandler(service=service, rabbitmq=rabbitmq)

    await worker_handler.embeddings_handle(
        _REPLY_TO,
        _CORRELATION_ID,
        {
            "action": "unexpected",
            "user_id": str(user_id),
            "texts": ["text"],
        },
    )

    payload = _last_reply_payload(rabbitmq)
    json.dumps(payload)
    assert payload["status"] == StatusType.ERROR.value


async def test_upsert_embedding_error_publishes_error_reply(user_id: uuid.UUID) -> None:
    """Base EmbeddingError from Triton must not escape the RabbitMQ handler."""
    service = MagicMock()
    service.upsert = AsyncMock(side_effect=EmbeddingError("triton unavailable"))
    rabbitmq = MagicMock()
    rabbitmq.publish_reply = AsyncMock()
    worker_handler = EmbeddingsWorkerHandler(service=service, rabbitmq=rabbitmq)

    await worker_handler.embeddings_handle(
        _REPLY_TO,
        _CORRELATION_ID,
        {
            "action": EmbeddingAction.UPSERT,
            "user_id": str(user_id),
            "texts": ["text"],
        },
    )

    payload = _last_reply_payload(rabbitmq)
    assert payload["status"] == StatusType.ERROR.value
    assert payload["details"] == "triton unavailable"


async def test_missing_reply_metadata_drops_message_without_reply(user_id: uuid.UUID) -> None:
    """Messages without reply metadata must not call publish_reply."""
    service = MagicMock()
    rabbitmq = MagicMock()
    rabbitmq.publish_reply = AsyncMock()
    worker_handler = EmbeddingsWorkerHandler(service=service, rabbitmq=rabbitmq)

    await worker_handler.embeddings_handle(
        reply_to=None,
        correlation_id=None,
        payload={
            "action": EmbeddingAction.UPSERT,
            "user_id": str(user_id),
            "texts": ["text"],
        },
    )

    rabbitmq.publish_reply.assert_not_awaited()
