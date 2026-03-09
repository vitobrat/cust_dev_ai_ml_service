import os
from enum import Enum
from pathlib import Path

_DEFAULT_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent

_DEFAULT_CONFIG_PATH = Path(_DEFAULT_PROJECT_PATH, "config", "config.dev.yaml")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", _DEFAULT_PROJECT_PATH))


class LogLevels(Enum):
    """Valid log level values accepted by uvicorn and the logging module."""

    DEBUG: str = "DEBUG"
    INFO: str = "INFO"  # noqa: WPS110
    WARNING: str = "WARNING"
    ERROR: str = "ERROR"
    CRITICAL: str = "CRITICAL"
