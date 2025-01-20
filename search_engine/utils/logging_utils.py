# utils/logging_utils.py

import logging
from functools import wraps
from ..config.config import get_config

# Configure logging
def configure_logger():
    """Configure the logger with settings from get_config."""
    config = get_config()
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    log_file = config.logging.file

    # Create logger
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        return logger

    # Set log level
    logger.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Initialize the logger globally
logger = configure_logger()

# Decorators
def log_execution(level=logging.INFO):
    """Logging decorator with configurable log level."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(level, f"Executing {func.__name__}...")
            result = func(*args, **kwargs)
            logger.log(level, f"Finished executing {func.__name__}")
            return result
        return wrapper
    return decorator


def handle_exceptions(func):
    """Exception handling decorator to capture and log errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return None
    return wrapper
