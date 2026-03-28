"""Integration tests for RabbitMQClient.

Tests cover connect/close lifecycle, publish/consume round-trip,
and publish_reply with a real RabbitMQ broker via testcontainers.
"""

import asyncio
import uuid
from typing import Any

from src.configs.config import RabbitMQConfigs
from src.infrastructure.rabbitmq.client import RabbitMQClient

_QUEUE_ID_LENGTH = 12
_CONSUME_TIMEOUT_SEC = 5.0
_BATCH_CONSUME_TIMEOUT_SEC = 10.0
_BATCH_MESSAGE_COUNT = 5


def _unique_queue() -> str:
    """Generate a unique queue name to isolate tests."""
    return f"test-queue-{uuid.uuid4().hex[:_QUEUE_ID_LENGTH]}"


class _PayloadCapture:
    """Capture the payload from a consumed message into an asyncio.Future."""

    def __init__(self, future: asyncio.Future[dict[str, Any]]) -> None:
        self._future = future

    async def __call__(
        self,
        reply_to: str | None,
        correlation_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        self._future.set_result(payload)


class _ReplyToCapture:
    """Capture reply_to from a consumed message into an asyncio.Future."""

    def __init__(self, future: asyncio.Future[str | None]) -> None:
        self._future = future

    async def __call__(
        self,
        reply_to: str | None,
        correlation_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        self._future.set_result(reply_to)


class _CorrelationIdCapture:
    """Capture correlation_id from a consumed message into an asyncio.Future."""

    def __init__(self, future: asyncio.Future[str | None]) -> None:
        self._future = future

    async def __call__(
        self,
        reply_to: str | None,
        correlation_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        self._future.set_result(correlation_id)


class _BatchCapture:
    """Collect payloads from multiple consumed messages."""

    def __init__(
        self,
        collected: list[dict[str, Any]],
        done_event: asyncio.Event,
        expected_count: int,
    ) -> None:
        self._collected = collected
        self._done_event = done_event
        self._expected_count = expected_count

    async def __call__(
        self,
        reply_to: str | None,
        correlation_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        self._collected.append(payload)
        if len(self._collected) >= self._expected_count:
            self._done_event.set()


class TestRabbitMQClientConnection:
    """Tests for connect() and close() lifecycle."""

    async def test_connect_establishes_connection(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that connect() sets _connection to a non-None value."""
        assert rabbitmq_client._connection is not None

    async def test_connect_creates_channel(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that connect() sets _channel to a non-None value."""
        assert rabbitmq_client._channel is not None

    async def test_close_releases_connection(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that close() sets _connection to None."""
        await rabbitmq_client.close()
        assert rabbitmq_client._connection is None

    async def test_close_releases_channel(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that close() sets _channel to None."""
        await rabbitmq_client.close()
        assert rabbitmq_client._channel is None

    async def test_close_without_connect_does_not_raise(
        self,
        rabbitmq_configs: RabbitMQConfigs,
    ) -> None:
        """Verify that close() without connect() does not raise."""
        client = RabbitMQClient(configs=rabbitmq_configs)
        await client.close()


class TestRabbitMQClientPublishConsume:
    """Tests for publish() and consume() round-trip."""

    async def test_publish_and_consume_receives_correct_payload(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that a published message is received with correct payload."""
        queue_name = _unique_queue()
        expected_payload = {"action": "test", "value": 42}
        received: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()

        await rabbitmq_client.consume(queue_name, _PayloadCapture(received))
        await rabbitmq_client.publish(queue_name, expected_payload)

        actual_payload = await asyncio.wait_for(received, timeout=_CONSUME_TIMEOUT_SEC)
        assert actual_payload == expected_payload

    async def test_publish_and_consume_reply_to_is_none_by_default(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that reply_to is None for a regular published message."""
        queue_name = _unique_queue()
        received: asyncio.Future[str | None] = asyncio.get_event_loop().create_future()

        await rabbitmq_client.consume(queue_name, _ReplyToCapture(received))
        await rabbitmq_client.publish(queue_name, {"ping": True})

        captured_reply_to = await asyncio.wait_for(received, timeout=_CONSUME_TIMEOUT_SEC)
        assert captured_reply_to is None

    async def test_publish_and_consume_correlation_id_is_none_by_default(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that correlation_id is None for a regular published message."""
        queue_name = _unique_queue()
        received: asyncio.Future[str | None] = asyncio.get_event_loop().create_future()

        await rabbitmq_client.consume(queue_name, _CorrelationIdCapture(received))
        await rabbitmq_client.publish(queue_name, {"ping": True})

        captured_corr_id = await asyncio.wait_for(received, timeout=_CONSUME_TIMEOUT_SEC)
        assert captured_corr_id is None

    async def test_publish_multiple_messages_all_consumed(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that multiple published messages are all consumed."""
        queue_name = _unique_queue()
        collected: list[dict[str, Any]] = []
        all_received = asyncio.Event()

        capture = _BatchCapture(collected, all_received, _BATCH_MESSAGE_COUNT)
        await rabbitmq_client.consume(queue_name, capture)

        for idx in range(_BATCH_MESSAGE_COUNT):
            await rabbitmq_client.publish(queue_name, {"index": idx})

        await asyncio.wait_for(all_received.wait(), timeout=_BATCH_CONSUME_TIMEOUT_SEC)
        assert len(collected) == _BATCH_MESSAGE_COUNT

    async def test_publish_to_nonexistent_queue_creates_it(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that publishing to a non-existent queue creates it automatically."""
        queue_name = _unique_queue()
        await rabbitmq_client.publish(queue_name, {"auto": "create"})

        received: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()

        await rabbitmq_client.consume(queue_name, _PayloadCapture(received))

        actual_payload = await asyncio.wait_for(received, timeout=_CONSUME_TIMEOUT_SEC)
        assert actual_payload == {"auto": "create"}


class TestRabbitMQClientPublishReplyIntegration:
    """Tests for publish_reply() with real broker."""

    async def test_publish_reply_delivers_to_reply_queue(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that publish_reply() delivers a message to the reply queue."""
        reply_queue = _unique_queue()
        expected_payload = {"result": "success"}
        received: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()

        await rabbitmq_client.consume(reply_queue, _PayloadCapture(received))
        await rabbitmq_client.publish_reply(reply_queue, "corr-abc", expected_payload)

        actual_payload = await asyncio.wait_for(received, timeout=_CONSUME_TIMEOUT_SEC)
        assert actual_payload == expected_payload

    async def test_publish_reply_preserves_correlation_id(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that publish_reply() preserves the correlation_id."""
        reply_queue = _unique_queue()
        expected_corr_id = "unique-corr-id-999"
        received: asyncio.Future[str | None] = asyncio.get_event_loop().create_future()

        await rabbitmq_client.consume(reply_queue, _CorrelationIdCapture(received))
        await rabbitmq_client.publish_reply(reply_queue, expected_corr_id, {"ok": True})

        actual_corr_id = await asyncio.wait_for(received, timeout=_CONSUME_TIMEOUT_SEC)
        assert actual_corr_id == expected_corr_id
