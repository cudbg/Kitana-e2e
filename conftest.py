import pytest
import logging
import os

@pytest.fixture(autouse=True)
def setup_logging():
    # Make sure logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)',
        handlers=[
            logging.FileHandler('logs/test.log'),
            logging.StreamHandler()
        ]
    )