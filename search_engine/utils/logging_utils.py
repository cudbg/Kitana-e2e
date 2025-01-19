# utils/logging_utils.py

import logging
from functools import wraps

# Configure logging with different levels
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
