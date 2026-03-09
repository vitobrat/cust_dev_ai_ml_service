import logging
import os
from logging.config import dictConfig
from pathlib import Path

import yaml


def setup_logger(logging_config_file: Path) -> None:
    """Setup logging configuration from a YAML file specified in settings."""
    if not os.path.exists(logging_config_file):
        logging.error(
            f"Unable to configure logging. {logging_config_file} not found",
        )
        return
    with open(logging_config_file, "r") as cfg_file:
        config = yaml.safe_load(cfg_file.read())
        dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
