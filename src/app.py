"""FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI, Request, Response, status
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
from src.infrastructure.health import build_readiness_payload
from src.schemas.api_base import ResponseBase, StatusType

settings = AppConfigs.init()

setup_logger(settings.logger.logging_config_file)
logger = get_logger(__name__)


def init_domain_containers() -> DomainContainer:
    """Initialize DI containers for app"""
    container = DomainContainer()
    container.config.from_dict(settings.model_dump())
    container.wire(packages=["src.domains"])

    return container


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initialises the DI container, wires all domain packages, and ensures
    the Qdrant collection exists on startup. RabbitMQ consumers are owned
    by ``src/worker.py``, not by the HTTP application lifespan.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the framework while the application is running.
    """
    container = init_domain_containers()
    app.state.container = container

    triton_client = container.infrastructure.triton_client()
    embedding_repo = container.embeddings.repository()

    await embedding_repo.ensure_collection()

    logger.info("ML service started successfully")

    try:
        yield
    finally:
        await triton_client.close()
        logger.info("ML service shut down successfully")


limiter = Limiter(key_func=get_remote_address)

api_v1_router = APIRouter(prefix="/api/v1")


@api_v1_router.get("/health/ready", response_model=ResponseBase)
async def readiness_probe(request: Request, response: Response) -> ResponseBase:
    """Return readiness of the external services required by the HTTP API."""
    container: DomainContainer = request.app.state.container
    payload = await build_readiness_payload(
        repository=container.embeddings.repository(),
        triton_client=container.infrastructure.triton_client(),
    )

    if payload["ready"]:
        return ResponseBase(msg=payload, status=StatusType.SUCCESS)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ResponseBase(msg=payload, status=StatusType.ERROR)


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
