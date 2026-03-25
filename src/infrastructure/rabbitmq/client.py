"""Async RabbitMQ client for publishing and consuming JSON messages."""

import json
from logging import Logger
from typing import Any, Protocol
from urllib.parse import quote

import aio_pika

from src.configs.config import RabbitMQConfigs
from src.configs.log.logger import get_logger


class MessageHandler(Protocol):
    """Protocol for RabbitMQ message handler callables.

    Any callable matching this signature can be used as a consumer handler.
    """

    async def __call__(
        self,
        reply_to: str | None,
        correlation_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        """Handle a single incoming RabbitMQ message.

        Args:
            reply_to: Queue name to publish the reply to, if provided.
            correlation_id: Opaque request ID to echo back in the reply.
            payload: Deserialised message body.
        """


class RabbitMQClient:
    """Async AMQP client for RabbitMQ message publishing and consumption.

    Manages a single robust connection and channel. The connection
    is created once at startup (via connect()) and reused for all
    operations. Reconnects automatically on network failures.

    Publish uses the default exchange with a routing key equal to
    the queue name. Consume declares the target queue as durable
    and wraps the user-supplied coroutine handler in an AMQP
    acknowledgment context.

    Attributes:
        _configs: RabbitMQ broker connection settings.
        _logger: Logger instance for this client.
        _connection: Active robust AMQP connection, or None before connect().
        _channel: Active AMQP channel, or None before connect().
    """

    _connection: aio_pika.abc.AbstractRobustConnection
    _channel: aio_pika.abc.AbstractChannel

    def __init__(self, configs: RabbitMQConfigs) -> None:
        """Store broker settings and initialise connection slots to None.

        Args:
            configs: RabbitMQ connection and credentials configuration.
        """
        self._configs = configs
        self._logger: Logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None

    async def connect(self) -> None:
        """Open a robust AMQP connection and create a channel.

        A robust connection automatically reconnects on network failures.
        Must be called once before any publish or consume operations.
        """
        self._logger.info("Connecting to RabbitMQ at %s:%s", self._configs.host, self._configs.port)
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()
        self._logger.info("RabbitMQ connection established")

    async def close(self) -> None:
        """Close the AMQP connection and release all resources.

        Safe to call even if connect() was never called.
        """
        if self._connection is None:
            return

        self._logger.info("Closing RabbitMQ connection")
        await self._connection.close()
        self._connection = None
        self._channel = None
        self._logger.info("RabbitMQ connection closed")

    async def publish(self, queue_name: str, payload: dict[str, Any]) -> None:
        """Publish a JSON-serialised payload to a queue via default exchange.

        Declares the target queue as durable before publishing so the
        queue is created automatically if it does not yet exist.

        Args:
            queue_name: Target queue name (also used as routing key).
            payload: Arbitrary JSON-serialisable dictionary to send.
        """
        self._logger.debug("Publishing message to queue '%s': %s", queue_name, payload)

        await self._channel.declare_queue(queue_name, durable=True)
        message = aio_pika.Message(body=json.dumps(payload).encode(), content_type="application/json")
        await self._channel.default_exchange.publish(message, routing_key=queue_name)

        self._logger.debug("Message published to queue '%s'", queue_name)

    async def publish_reply(
        self,
        reply_to: str,
        correlation_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Publish a JSON reply to a client-owned reply queue.

        Unlike publish(), skips queue declaration because the reply queue
        is created by the caller and may be transient. Attaches correlation_id
        so the caller can match this response to the original request.

        Args:
            reply_to: Target reply queue name (provided by the original message).
            correlation_id: Opaque ID from the original request to echo back.
            payload: Arbitrary JSON-serialisable dictionary to send as the reply.
        """
        self._logger.debug("Publishing reply to queue '%s': %s", reply_to, payload)

        message = aio_pika.Message(
            body=json.dumps(payload).encode(),
            content_type="application/json",
            correlation_id=correlation_id,
        )
        await self._channel.default_exchange.publish(message, routing_key=reply_to)

        self._logger.debug("Reply published to queue '%s'", reply_to)

    async def consume(self, queue_name: str, consume_handler: MessageHandler) -> None:
        """Start consuming messages from a queue with a coroutine handler.

        Declares the queue as durable, then registers an async consumer.
        Each incoming message is auto-acknowledged after the handler
        completes without raising an exception.

        Args:
            queue_name: Queue to consume from (declared as durable).
            consume_handler: Async coroutine receiving reply_to, correlation_id, and payload.
        """
        self._logger.info("Registering consumer for queue '%s'", queue_name)
        queue = await self._channel.declare_queue(queue_name, durable=True)

        await queue.consume(
            lambda msg: self._consume_handle(msg, consume_handler, queue_name),
        )
        self._logger.info("Consumer registered for queue '%s'", queue_name)

    @property
    def _url(self) -> str:
        """Build an AMQP connection URL from stored configs.

        Returns:
            AMQP URL string in format: amqp://user:password@host:port/vhost
        """
        vhost = quote(self._configs.vhost, safe="")
        return (
            f"amqp://{self._configs.user}:{self._configs.password}"
            f"@{self._configs.host}:{self._configs.port}/{vhost}"
        )

    async def _consume_handle(
        self,
        message: aio_pika.abc.AbstractIncomingMessage,
        consume_handler: MessageHandler,
        queue_name: str,
    ) -> None:
        """Process a single incoming message from a queue.

        Acknowledges the message after the handler completes
        without raising an exception.

        Args:
            message: Incoming AMQP message to process.
            consume_handler: Async coroutine receiving reply_to, correlation_id, and payload.
            queue_name: Name of the source queue (used for logging).
        """
        async with message.process():
            reply_to = message.reply_to
            correlation_id = message.correlation_id
            payload = json.loads(message.body)
            self._logger.debug("Received message from queue '%s': %s", queue_name, payload)

            await consume_handler(
                reply_to,
                correlation_id,
                payload,
            )
