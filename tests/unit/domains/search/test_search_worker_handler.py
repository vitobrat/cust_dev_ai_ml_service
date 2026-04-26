"""Unit tests for SearchWorkerHandler."""

import json
import uuid
from typing import TypedDict, cast
from unittest.mock import AsyncMock, MagicMock

from src.domains.search.app.workers.handler import SearchWorkerHandler
from src.infrastructure.exceptions import EmbeddingError
from src.schemas.api_base import StatusType

_REPLY_TO = "reply-queue"
_CORRELATION_ID = "corr-123"


class _SearchResultPayload(TypedDict):
    """Search result item payload in a serialized worker reply."""

    id: str


class _SearchSuccessReply(TypedDict):
    """Successful search worker reply payload."""

    status: str
    msg: list[_SearchResultPayload]


def _last_reply_payload(rabbitmq: MagicMock) -> dict[str, object]:
    """Return the last payload passed to publish_reply()."""
    return rabbitmq.publish_reply.call_args.kwargs["payload"]


async def test_search_success_reply_is_json_serializable(user_id: uuid.UUID) -> None:
    """Successful search replies must contain JSON primitives, not Enum or UUID objects."""
    result_id = uuid.uuid4()
    result_item = {"id": result_id, "score": 0.9, "payload": {"text": "doc"}}
    service = MagicMock()
    service.search = AsyncMock(return_value=[result_item])
    rabbitmq = MagicMock()
    rabbitmq.publish_reply = AsyncMock()
    worker_handler = SearchWorkerHandler(service=service, rabbitmq=rabbitmq)

    await worker_handler.search_handle(
        _REPLY_TO,
        _CORRELATION_ID,
        {"user_id": str(user_id), "query": "query", "top_k": 1},
    )

    payload = _last_reply_payload(rabbitmq)
    success_reply = cast(_SearchSuccessReply, payload)
    json.dumps(payload)
    assert success_reply["status"] == StatusType.SUCCESS.value
    assert success_reply["msg"][0]["id"] == str(result_id)


async def test_invalid_search_payload_error_reply_is_json_serializable() -> None:
    """Validation errors must be returned as JSON-serializable replies."""
    service = MagicMock()
    rabbitmq = MagicMock()
    rabbitmq.publish_reply = AsyncMock()
    worker_handler = SearchWorkerHandler(service=service, rabbitmq=rabbitmq)

    await worker_handler.search_handle(_REPLY_TO, _CORRELATION_ID, {"query": "missing user_id"})

    payload = _last_reply_payload(rabbitmq)
    json.dumps(payload)
    assert payload["status"] == StatusType.ERROR.value


async def test_search_embedding_error_publishes_error_reply(user_id: uuid.UUID) -> None:
    """Base EmbeddingError from Triton must not escape the RabbitMQ handler."""
    service = MagicMock()
    service.search = AsyncMock(side_effect=EmbeddingError("triton unavailable"))
    rabbitmq = MagicMock()
    rabbitmq.publish_reply = AsyncMock()
    worker_handler = SearchWorkerHandler(service=service, rabbitmq=rabbitmq)

    await worker_handler.search_handle(
        _REPLY_TO,
        _CORRELATION_ID,
        {"user_id": str(user_id), "query": "query", "top_k": 1},
    )

    payload = _last_reply_payload(rabbitmq)
    assert payload["status"] == StatusType.ERROR.value
    assert payload["details"] == "triton unavailable"


async def test_missing_reply_metadata_drops_message_without_reply(user_id: uuid.UUID) -> None:
    """Messages without reply metadata must not call publish_reply."""
    service = MagicMock()
    rabbitmq = MagicMock()
    rabbitmq.publish_reply = AsyncMock()
    worker_handler = SearchWorkerHandler(service=service, rabbitmq=rabbitmq)

    await worker_handler.search_handle(
        reply_to=None,
        correlation_id=None,
        payload={"user_id": str(user_id), "query": "query", "top_k": 1},
    )

    rabbitmq.publish_reply.assert_not_awaited()
