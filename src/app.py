"""FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.configs.config import AppConfigs
from src.configs.log.logger import get_logger, setup_logger
from src.domains.embeddings.app.requests.router import (
    router as embeddings_router,
)
from src.domains.search.app.requests.router import router as search_router
from src.infrastructure.containers.domain import DomainContainer

settings = AppConfigs.init()

setup_logger(settings.logger.logging_config_file)
logger = get_logger(__name__)


def init_containers() -> DomainContainer:
    """Initialize DI containers for app"""
    container = DomainContainer()
    container.config.from_dict(settings.model_dump())
    container.wire(packages=["src.domains"])

    return container


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initialises the DI container, wires all domain packages, connects to
    RabbitMQ, and ensures the Qdrant collection exists on startup. Closes
    all connections gracefully on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the framework while the application is running.
    """
    container = init_containers()

    rabbitmq_client = container.infrastructure.rabbitmq_client()
    triton_client = container.infrastructure.triton_client()
    embedding_repo = container.embeddings.repository()

    await rabbitmq_client.connect()
    await embedding_repo.ensure_collection()

    logger.info("ML service started successfully")

    try:  # noqa: WPS243
        yield
    finally:
        await triton_client.close()
        await rabbitmq_client.close()
        logger.info("ML service shut down successfully")


limiter = Limiter(key_func=get_remote_address)

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(embeddings_router)
api_v1_router.include_router(search_router)

app = FastAPI(lifespan=lifespan)
app.include_router(api_v1_router)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.value.lower(),
        workers=settings.workers_number,
    )
