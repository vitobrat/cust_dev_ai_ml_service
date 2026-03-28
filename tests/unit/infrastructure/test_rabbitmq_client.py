"""Unit tests for RabbitMQClient.

Tests cover all public methods of the async RabbitMQ client
with mocked aio_pika dependencies. No real broker connection is used.
"""

import json
from unittest.mock import AsyncMock, patch

from src.configs.config import RabbitMQConfigs
from src.infrastructure.rabbitmq.client import RabbitMQClient


class TestRabbitMQClientInit:
    """Tests for RabbitMQClient.__init__()."""

    def test_init_stores_configs(
        self,
        rabbitmq_configs: RabbitMQConfigs,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that __init__ stores the provided configs."""
        assert rabbitmq_client._configs is rabbitmq_configs

    def test_init_connection_is_none(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that __init__ sets _connection to None."""
        assert rabbitmq_client._connection is None

    def test_init_channel_is_none(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that __init__ sets _channel to None."""
        assert rabbitmq_client._channel is None


class TestRabbitMQClientUrl:
    """Tests for RabbitMQClient._url property."""

    def test_url_starts_with_amqp_scheme(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that _url starts with 'amqp://'."""
        assert rabbitmq_client._url.startswith("amqp://")

    def test_url_contains_host_and_port(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that _url contains host:port."""
        assert "localhost:5672" in rabbitmq_client._url

    def test_url_contains_credentials(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that _url contains user:password."""
        assert "guest:guest@" in rabbitmq_client._url

    def test_url_encodes_vhost_with_slash(
        self,
        rabbitmq_configs: RabbitMQConfigs,
    ) -> None:
        """Verify that vhost '/' is URL-encoded to '%2F'."""
        rabbitmq_configs.vhost = "/"
        client = RabbitMQClient(configs=rabbitmq_configs)
        assert client._url.endswith("%2F")

    def test_url_encodes_special_chars_in_vhost(
        self,
        rabbitmq_configs: RabbitMQConfigs,
    ) -> None:
        """Verify that special characters in vhost are URL-encoded."""
        rabbitmq_configs.vhost = "my/vhost"
        client = RabbitMQClient(configs=rabbitmq_configs)
        assert client._url.endswith("my%2Fvhost")

    def test_url_format_matches_amqp_spec(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify the full URL format: amqp://user:pass@host:port/vhost."""
        expected = "amqp://guest:guest@localhost:5672/%2F"
        assert rabbitmq_client._url == expected

    def test_url_plain_vhost_not_encoded(
        self,
        rabbitmq_configs: RabbitMQConfigs,
    ) -> None:
        """Verify that a plain alphanumeric vhost is not encoded."""
        rabbitmq_configs.vhost = "production"
        client = RabbitMQClient(configs=rabbitmq_configs)
        assert client._url.endswith("/production")


class TestRabbitMQClientConnect:
    """Tests for RabbitMQClient.connect()."""

    @patch("src.infrastructure.rabbitmq.client.aio_pika.connect_robust")
    async def test_connect_calls_connect_robust_with_url(
        self,
        mock_connect_robust: AsyncMock,
        rabbitmq_client: RabbitMQClient,
        mock_connection: AsyncMock,
    ) -> None:
        """Verify that connect() calls aio_pika.connect_robust with correct URL."""
        mock_connect_robust.return_value = mock_connection
        await rabbitmq_client.connect()
        mock_connect_robust.assert_awaited_once_with(rabbitmq_client._url)

    @patch("src.infrastructure.rabbitmq.client.aio_pika.connect_robust")
    async def test_connect_creates_channel(
        self,
        mock_connect_robust: AsyncMock,
        rabbitmq_client: RabbitMQClient,
        mock_connection: AsyncMock,
    ) -> None:
        """Verify that connect() creates a channel from the connection."""
        mock_connect_robust.return_value = mock_connection
        await rabbitmq_client.connect()
        mock_connection.channel.assert_awaited_once()

    @patch("src.infrastructure.rabbitmq.client.aio_pika.connect_robust")
    async def test_connect_stores_connection(
        self,
        mock_connect_robust: AsyncMock,
        rabbitmq_client: RabbitMQClient,
        mock_connection: AsyncMock,
    ) -> None:
        """Verify that connect() stores the connection on the client."""
        mock_connect_robust.return_value = mock_connection
        await rabbitmq_client.connect()
        assert rabbitmq_client._connection is mock_connection

    @patch("src.infrastructure.rabbitmq.client.aio_pika.connect_robust")
    async def test_connect_stores_channel(
        self,
        mock_connect_robust: AsyncMock,
        rabbitmq_client: RabbitMQClient,
        mock_connection: AsyncMock,
        mock_channel: AsyncMock,
    ) -> None:
        """Verify that connect() stores the channel on the client."""
        mock_connect_robust.return_value = mock_connection
        await rabbitmq_client.connect()
        assert rabbitmq_client._channel is mock_channel


class TestRabbitMQClientClose:
    """Tests for RabbitMQClient.close()."""

    async def test_close_when_not_connected_does_nothing(
        self,
        rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that close() is safe when _connection is None."""
        await rabbitmq_client.close()
        assert rabbitmq_client._connection is None

    async def test_close_calls_connection_close(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_connection: AsyncMock,
    ) -> None:
        """Verify that close() calls connection.close()."""
        await connected_rabbitmq_client.close()
        mock_connection.close.assert_awaited_once()

    async def test_close_sets_connection_to_none(
        self,
        connected_rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that close() sets _connection to None."""
        await connected_rabbitmq_client.close()
        assert connected_rabbitmq_client._connection is None

    async def test_close_sets_channel_to_none(
        self,
        connected_rabbitmq_client: RabbitMQClient,
    ) -> None:
        """Verify that close() sets _channel to None."""
        await connected_rabbitmq_client.close()
        assert connected_rabbitmq_client._channel is None


class TestRabbitMQClientPublish:
    """Tests for RabbitMQClient.publish()."""

    async def test_publish_declares_durable_queue(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_channel: AsyncMock,
    ) -> None:
        """Verify that publish() declares a durable queue."""
        await connected_rabbitmq_client.publish("test-queue", {"key": "value"})
        mock_channel.declare_queue.assert_awaited_once_with("test-queue", durable=True)

    async def test_publish_sends_message_to_default_exchange(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_default_exchange: AsyncMock,
    ) -> None:
        """Verify that publish() sends a message via the default exchange."""
        await connected_rabbitmq_client.publish("test-queue", {"key": "value"})
        mock_default_exchange.publish.assert_awaited_once()

    async def test_publish_uses_queue_name_as_routing_key(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_default_exchange: AsyncMock,
    ) -> None:
        """Verify that publish() uses queue_name as routing_key."""
        queue_name = "my-queue"
        await connected_rabbitmq_client.publish(queue_name, {"data": 1})

        call_kwargs = mock_default_exchange.publish.call_args
        assert call_kwargs.kwargs["routing_key"] == queue_name

    async def test_publish_message_body_is_json_encoded(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_default_exchange: AsyncMock,
    ) -> None:
        """Verify that the message body is JSON-encoded bytes."""
        payload = {"key": "value", "number": 42}
        await connected_rabbitmq_client.publish("test-queue", payload)

        call_args = mock_default_exchange.publish.call_args
        message = call_args.args[0]
        assert message.body == json.dumps(payload).encode()

    async def test_publish_message_content_type_is_json(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_default_exchange: AsyncMock,
    ) -> None:
        """Verify that the message content_type is application/json."""
        await connected_rabbitmq_client.publish("test-queue", {"key": "value"})

        call_args = mock_default_exchange.publish.call_args
        message = call_args.args[0]
        assert message.content_type == "application/json"


class TestRabbitMQClientPublishReply:
    """Tests for RabbitMQClient.publish_reply()."""

    async def test_publish_reply_does_not_declare_queue(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_channel: AsyncMock,
    ) -> None:
        """Verify that publish_reply() skips queue declaration."""
        await connected_rabbitmq_client.publish_reply("reply-q", "corr-1", {"ok": True})
        mock_channel.declare_queue.assert_not_awaited()

    async def test_publish_reply_sends_to_reply_to_queue(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_default_exchange: AsyncMock,
    ) -> None:
        """Verify that publish_reply() uses reply_to as routing_key."""
        reply_to = "client-reply-queue"
        await connected_rabbitmq_client.publish_reply(reply_to, "corr-1", {"ok": True})

        call_kwargs = mock_default_exchange.publish.call_args
        assert call_kwargs.kwargs["routing_key"] == reply_to

    async def test_publish_reply_sets_correlation_id(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_default_exchange: AsyncMock,
    ) -> None:
        """Verify that publish_reply() sets correlation_id on the message."""
        correlation_id = "request-abc-123"
        await connected_rabbitmq_client.publish_reply("reply-q", correlation_id, {"ok": True})

        call_args = mock_default_exchange.publish.call_args
        message = call_args.args[0]
        assert message.correlation_id == correlation_id

    async def test_publish_reply_body_is_json_encoded(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_default_exchange: AsyncMock,
    ) -> None:
        """Verify that the reply message body is JSON-encoded bytes."""
        payload = {"status": "done", "count": 5}
        await connected_rabbitmq_client.publish_reply("reply-q", "corr-1", payload)

        call_args = mock_default_exchange.publish.call_args
        message = call_args.args[0]
        assert message.body == json.dumps(payload).encode()

    async def test_publish_reply_content_type_is_json(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_default_exchange: AsyncMock,
    ) -> None:
        """Verify that the reply message content_type is application/json."""
        await connected_rabbitmq_client.publish_reply("reply-q", "corr-1", {"ok": True})

        call_args = mock_default_exchange.publish.call_args
        message = call_args.args[0]
        assert message.content_type == "application/json"


class TestRabbitMQClientConsume:
    """Tests for RabbitMQClient.consume()."""

    async def test_consume_declares_durable_queue(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_channel: AsyncMock,
    ) -> None:
        """Verify that consume() declares a durable queue."""
        consume_callback = AsyncMock()
        await connected_rabbitmq_client.consume("input-queue", consume_callback)
        mock_channel.declare_queue.assert_awaited_once_with("input-queue", durable=True)

    async def test_consume_registers_consumer_on_queue(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_queue: AsyncMock,
    ) -> None:
        """Verify that consume() calls queue.consume()."""
        consume_callback = AsyncMock()
        await connected_rabbitmq_client.consume("input-queue", consume_callback)
        mock_queue.consume.assert_awaited_once()

    async def test_consume_passes_callable_to_queue_consume(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_queue: AsyncMock,
    ) -> None:
        """Verify that the first argument to queue.consume is a callable."""
        consume_callback = AsyncMock()
        await connected_rabbitmq_client.consume("input-queue", consume_callback)

        call_args = mock_queue.consume.call_args
        registered_fn = call_args.args[0]
        assert callable(registered_fn)


class TestRabbitMQClientConsumeHandle:
    """Tests for RabbitMQClient._consume_handle() internal method."""

    async def test_consume_handle_calls_handler(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_incoming_message: AsyncMock,
    ) -> None:
        """Verify that _consume_handle calls the handler."""
        consume_callback = AsyncMock()
        await connected_rabbitmq_client._consume_handle(
            mock_incoming_message,
            consume_callback,
            "test-queue",
        )
        consume_callback.assert_awaited_once()

    async def test_consume_handle_passes_reply_to_from_message(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_incoming_message: AsyncMock,
    ) -> None:
        """Verify that _consume_handle passes reply_to from the message."""
        consume_callback = AsyncMock()
        await connected_rabbitmq_client._consume_handle(
            mock_incoming_message,
            consume_callback,
            "test-queue",
        )
        call_args = consume_callback.call_args
        assert call_args.args[0] == "reply-queue"

    async def test_consume_handle_passes_correlation_id_from_message(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_incoming_message: AsyncMock,
    ) -> None:
        """Verify that _consume_handle passes correlation_id from the message."""
        consume_callback = AsyncMock()
        await connected_rabbitmq_client._consume_handle(
            mock_incoming_message,
            consume_callback,
            "test-queue",
        )
        call_args = consume_callback.call_args
        assert call_args.args[1] == "corr-123"

    async def test_consume_handle_deserializes_json_body(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_incoming_message: AsyncMock,
    ) -> None:
        """Verify that _consume_handle deserializes JSON body into dict."""
        consume_callback = AsyncMock()
        await connected_rabbitmq_client._consume_handle(
            mock_incoming_message,
            consume_callback,
            "test-queue",
        )
        call_args = consume_callback.call_args
        assert call_args.args[2] == {"key": "value"}

    async def test_consume_handle_uses_message_process_context(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_incoming_message: AsyncMock,
    ) -> None:
        """Verify that _consume_handle uses message.process() as context manager."""
        consume_callback = AsyncMock()
        await connected_rabbitmq_client._consume_handle(
            mock_incoming_message,
            consume_callback,
            "test-queue",
        )
        mock_incoming_message.process.assert_called_once()

    async def test_consume_handle_with_none_reply_to(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_incoming_message: AsyncMock,
    ) -> None:
        """Verify that _consume_handle passes None reply_to when absent."""
        mock_incoming_message.reply_to = None
        consume_callback = AsyncMock()
        await connected_rabbitmq_client._consume_handle(
            mock_incoming_message,
            consume_callback,
            "test-queue",
        )
        call_args = consume_callback.call_args
        assert call_args.args[0] is None

    async def test_consume_handle_with_none_correlation_id(
        self,
        connected_rabbitmq_client: RabbitMQClient,
        mock_incoming_message: AsyncMock,
    ) -> None:
        """Verify that _consume_handle passes None correlation_id when absent."""
        mock_incoming_message.correlation_id = None
        consume_callback = AsyncMock()
        await connected_rabbitmq_client._consume_handle(
            mock_incoming_message,
            consume_callback,
            "test-queue",
        )
        call_args = consume_callback.call_args
        assert call_args.args[1] is None
