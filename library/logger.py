"""
Module where a logger instance is configured to be used at the service.
"""
import os
import sys

from loguru import logger


log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
log_format = "<green>{time}</green> - <level>{level}</level>: {message}"

logger.remove(handler_id=0)
logger.add(sink="logs/churn_library.log", format=log_format, level=log_level, colorize=True)
