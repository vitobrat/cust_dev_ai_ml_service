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

settings = AppConfigs.init()

setup_logger(settings.logger.logging_config_file)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initializes the DI container, wires all domain packages on startup,
    and logs a shutdown message on teardown.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the framework while the application is running.
    """
    logger.info("ML service is starting up...")

    try:
        yield
    finally:
        logger.info("Service is shutting down correctly")


limiter = Limiter(key_func=get_remote_address)

api_v1_router = APIRouter(prefix="/api/v1")

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
