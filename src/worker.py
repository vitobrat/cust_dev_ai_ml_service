"""Worker entry point: consumes RabbitMQ messages and dispatches to domain handlers."""

import asyncio
import signal

from src.configs.config import AppConfigs
from src.configs.consts import EMBEDDINGS_RABBITMQ_QUEUE, SEARCH_RABBITMQ_QUEUE
from src.configs.log.logger import get_logger, setup_logger
from src.infrastructure.containers.domain import DomainContainer
from src.infrastructure.rabbitmq.client import RabbitMQClient
from src.infrastructure.triton.client import TritonClient

settings = AppConfigs.init()

setup_logger(settings.logger.logging_config_file)
logger = get_logger(__name__)


def init_containers() -> DomainContainer:
    """Initialize and wire DI containers for the worker."""
    container = DomainContainer()
    container.config.from_dict(settings.model_dump())
    container.wire(packages=["src.domains"])
    return container


async def _register_consumers(container: DomainContainer, rabbitmq_client: RabbitMQClient) -> None:
    """Register RabbitMQ consumers for all domain handlers.

    Args:
        container: Wired DI container providing domain handlers.
        rabbitmq_client: Connected RabbitMQ client to register consumers on.
    """
    await rabbitmq_client.consume(
        EMBEDDINGS_RABBITMQ_QUEUE,
        container.embeddings.embeddings_handler().embeddings_handle,  # type: ignore[operator]
    )
    await rabbitmq_client.consume(
        SEARCH_RABBITMQ_QUEUE,
        container.search.search_handler().search_handle,  # type: ignore[operator]
    )


def _setup_stop_event() -> asyncio.Event:
    """Create an Event and bind SIGINT/SIGTERM to set it.

    Returns:
        Event that will be set when a shutdown signal is received.
    """
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop_event.set)
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    return stop_event


async def _shutdown(triton_client: TritonClient, rabbitmq_client: RabbitMQClient) -> None:
    """Close all external service connections gracefully.

    Args:
        triton_client: Triton gRPC client to close.
        rabbitmq_client: RabbitMQ client to close.
    """
    await triton_client.close()
    await rabbitmq_client.close()
    logger.info("Worker shut down successfully")


async def main() -> None:
    """Run the worker: connect to services, register consumers, and wait for shutdown.

    Initialises the DI container, connects to RabbitMQ, ensures the Qdrant
    collection exists, registers message consumers, and blocks until a
    SIGINT or SIGTERM signal is received. Closes all connections on exit.
    """
    container = init_containers()

    rabbitmq_client = container.infrastructure.rabbitmq_client()
    triton_client = container.infrastructure.triton_client()
    embedding_repo = container.embeddings.repository()

    await rabbitmq_client.connect()
    await embedding_repo.ensure_collection()
    await _register_consumers(container, rabbitmq_client)

    logger.info("Worker started. Listening on '%s' and '%s'", EMBEDDINGS_RABBITMQ_QUEUE, SEARCH_RABBITMQ_QUEUE)

    stop_event = _setup_stop_event()

    try:
        await stop_event.wait()
    except Exception as exc:
        logger.error("Unexpected worker error: %s", exc)
    finally:
        await _shutdown(triton_client, rabbitmq_client)


if __name__ == "__main__":
    asyncio.run(main())
