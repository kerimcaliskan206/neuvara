import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def log_duration(operation: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    logger.info("[ML] Starting: %s", operation)
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("[ML] Finished: %s in %.1fms", operation, elapsed)


def log_step(step_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info("[ML Step] %s → started", step_name)
            result = func(*args, **kwargs)
            logger.info("[ML Step] %s → done", step_name)
            return result
        return wrapper
    return decorator
