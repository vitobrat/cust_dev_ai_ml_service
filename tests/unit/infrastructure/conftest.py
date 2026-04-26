"""Fixtures for infrastructure-layer unit tests."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from aio_pika.abc import (
    AbstractChannel,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
)
from qdrant_client import AsyncQdrantClient

from src.configs.config import RabbitMQConfigs
from src.infrastructure.db.qdrant.embedding_repository import (
    EmbeddingRepository,
)
from src.infrastructure.rabbitmq.client import RabbitMQClient

_TEST_COLLECTION: str = "test_documents"
_TEST_VECTOR_SIZE: int = 5
_TEST_HNSW_M: int = 4
_TEST_HNSW_EF: int = 16


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Return an AsyncQdrantClient mock with all async operations pre-configured.

    Each method is an AsyncMock so it can be awaited inside repository methods.
    collection_exists defaults to False so ensure_collection triggers creation.

    Returns:
        MagicMock configured to act as an AsyncQdrantClient.
    """
    mock = MagicMock(spec=AsyncQdrantClient)
    mock.collection_exists = AsyncMock(return_value=False)
    mock.create_collection = AsyncMock()
    mock.create_payload_index = AsyncMock()
    collection_info = MagicMock()
    collection_info.payload_schema = {}
    mock.get_collection = AsyncMock(return_value=collection_info)
    mock.upsert = AsyncMock()
    mock.delete = AsyncMock()
    mock_result = MagicMock()
    mock_result.points = []
    mock.query_points = AsyncMock(return_value=mock_result)
    return mock


@pytest.fixture
def repository(mock_qdrant_client: MagicMock) -> EmbeddingRepository:
    """Return EmbeddingRepository wired with a mocked Qdrant client.

    Args:
        mock_qdrant_client: Mocked AsyncQdrantClient.

    Returns:
        EmbeddingRepository ready for isolated unit testing.
    """
    return EmbeddingRepository(
        client=cast(AsyncQdrantClient, mock_qdrant_client),
        collection_name=_TEST_COLLECTION,
        vector_size=_TEST_VECTOR_SIZE,
        hnsw_edge_size=_TEST_HNSW_M,
        hnsw_neighbour_size=_TEST_HNSW_EF,
    )


@pytest.fixture
def rabbitmq_configs() -> RabbitMQConfigs:
    """Provide a mock RabbitMQConfigs instance with test values."""
    configs = MagicMock(spec=RabbitMQConfigs)
    configs.host = "localhost"
    configs.port = 5672
    configs.vhost = "/"
    configs.user = "guest"
    configs.password = "guest"
    return configs


@pytest.fixture
def mock_default_exchange() -> AsyncMock:
    """Provide a mock AMQP default exchange."""
    exchange = AsyncMock()
    exchange.publish = AsyncMock()
    return exchange


@pytest.fixture
def mock_queue() -> AsyncMock:
    """Provide a mock AMQP queue."""
    queue = AsyncMock(spec=AbstractQueue)
    queue.consume = AsyncMock()
    return queue


@pytest.fixture
def mock_channel(
    mock_default_exchange: AsyncMock,
    mock_queue: AsyncMock,
) -> AsyncMock:
    """Provide a mock AMQP channel with default exchange and declare_queue."""
    channel = AsyncMock(spec=AbstractChannel)
    channel.default_exchange = mock_default_exchange
    channel.declare_queue = AsyncMock(return_value=mock_queue)
    return channel


@pytest.fixture
def mock_connection(mock_channel: AsyncMock) -> AsyncMock:
    """Provide a mock robust AMQP connection returning mock_channel."""
    connection = AsyncMock(spec=AbstractRobustConnection)
    connection.channel = AsyncMock(return_value=mock_channel)
    connection.close = AsyncMock()
    return connection


@pytest.fixture
def rabbitmq_client(rabbitmq_configs: RabbitMQConfigs) -> RabbitMQClient:
    """Provide a fresh RabbitMQClient without active connection."""
    return RabbitMQClient(configs=rabbitmq_configs)


@pytest.fixture
def connected_rabbitmq_client(
    rabbitmq_client: RabbitMQClient,
    mock_connection: AsyncMock,
    mock_channel: AsyncMock,
) -> RabbitMQClient:
    """Provide a RabbitMQClient with pre-injected mock connection and channel."""
    rabbitmq_client._connection = mock_connection
    rabbitmq_client._channel = mock_channel
    return rabbitmq_client


@pytest.fixture
def mock_incoming_message() -> AsyncMock:
    """Provide a mock incoming AMQP message with process() context manager."""
    message = AsyncMock(spec=AbstractIncomingMessage)
    message.reply_to = "reply-queue"
    message.correlation_id = "corr-123"
    message.body = b'{"key": "value"}'

    process_cm = AsyncMock()
    process_cm.__aenter__ = AsyncMock(return_value=None)
    process_cm.__aexit__ = AsyncMock(return_value=False)
    message.process = MagicMock(return_value=process_cm)

    return message
