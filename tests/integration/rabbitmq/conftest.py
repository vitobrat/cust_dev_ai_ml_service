"""Pytest fixtures for integration RabbitMQ tests.

This module provides a RabbitMQ testcontainer and a connected
RabbitMQClient fixture for each test function.
"""

from collections.abc import AsyncGenerator
from typing import Generator

import pytest
import pytest_asyncio
from testcontainers.rabbitmq import RabbitMqContainer

from src.configs.config import RabbitMQConfigs
from src.infrastructure.rabbitmq.client import RabbitMQClient

_RABBITMQ_IMAGE = "rabbitmq:3.13-alpine"
_RABBITMQ_PORT = 5672


@pytest.fixture(scope="session")
def rabbitmq_container() -> Generator[RabbitMqContainer, None, None]:
    """Provide a RabbitMQ testcontainer running for the entire test session.

    Yields:
        Started RabbitMqContainer instance.
    """
    with RabbitMqContainer(
        image=_RABBITMQ_IMAGE,
        port=_RABBITMQ_PORT,
        username="guest",
        password="guest",
    ) as container:
        yield container


@pytest.fixture(scope="session")
def rabbitmq_configs(rabbitmq_container: RabbitMqContainer) -> RabbitMQConfigs:
    """Build RabbitMQConfigs from the running testcontainer.

    Args:
        rabbitmq_container: Running RabbitMQ testcontainer.

    Returns:
        RabbitMQConfigs with host, mapped port, and credentials.
    """
    return RabbitMQConfigs(
        host=rabbitmq_container.get_container_host_ip(),
        port=int(rabbitmq_container.get_exposed_port(_RABBITMQ_PORT)),
        vhost="/",
        RABBITMQ_USER="guest",
        RABBITMQ_PASSWORD="guest",
    )


@pytest_asyncio.fixture(scope="function")
async def rabbitmq_client(
    rabbitmq_configs: RabbitMQConfigs,
) -> AsyncGenerator[RabbitMQClient, None]:
    """Provide a connected RabbitMQClient for each test, closed after use.

    Args:
        rabbitmq_configs: Connection settings from the testcontainer.

    Yields:
        Connected RabbitMQClient instance.
    """
    client = RabbitMQClient(configs=rabbitmq_configs)
    await client.connect()
    yield client
    await client.close()
