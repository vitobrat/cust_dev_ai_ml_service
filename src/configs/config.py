"""Configuration module for the application.

This module provides Pydantic-based configuration classes for managing
application settings from YAML files and environment variables. It supports
nested configurations and automatic validation.
"""

import os
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.configs.consts import _DEFAULT_CONFIG_PATH, PROJECT_ROOT, LogLevels
from src.configs.log.logger import get_logger

dotenv_path = Path(PROJECT_ROOT, "config", ".env")
logger = get_logger(__name__)


class _BaseValidatedConfig(BaseSettings):
    """Base configuration class with validation and environment variable support.

    This class provides common configuration for all settings classes,
    including environment file loading and nested delimiter support.

    Attributes:
        model_config: Pydantic settings configuration with env file path and encoding.
    """

    model_config = SettingsConfigDict(
        env_file=str(dotenv_path),
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="_",
    )


class LoggerConfigs(_BaseValidatedConfig):
    """Logging configuration settings.

    Attributes:
        logging_config_file: Path to the logging configuration YAML file.
    """

    logging_config_file: Path = Path(PROJECT_ROOT, "config", "logging.yaml")


class AppConfigs(_BaseValidatedConfig):
    """Root application configuration aggregating all subsystem configs.

    This class serves as the main entry point for application configuration,
    combining all domain and infrastructure settings.

    Attributes:
        app_host: Application server host address.
        app_port: Application server port number.
        log_level: Logging verbosity level.
        workers_number: Number of uvicorn worker processes.
        logger: Logging subsystem configuration.
    """

    app_host: str
    app_port: int
    log_level: LogLevels
    workers_number: int
    logger: LoggerConfigs = Field(default_factory=LoggerConfigs)

    @classmethod
    def init(cls) -> "AppConfigs":
        """Initialize application configuration from YAML file.

        Loads configuration from YAML file specified by CONFIG_PATH environment
        variable, or uses default path if not set. Resolves OmegaConf interpolations
        and validates all settings.

        Returns:
            Fully initialized and validated AppConfigs instance.

        Raises:
            ValidationError: If configuration values fail Pydantic validation.
            FileNotFoundError: If configuration file does not exist.
        """
        config_path_str = os.getenv("CONFIG_PATH")
        path = Path(config_path_str) if config_path_str else _DEFAULT_CONFIG_PATH
        yaml_config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

        return cls(**yaml_config)
