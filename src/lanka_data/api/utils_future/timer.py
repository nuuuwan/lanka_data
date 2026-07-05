import functools
import os
import time
from typing import Any

from lanka_data.api.utils_future.Log import Log

log = Log("timer")


def _get_logger(elapsed):
    if elapsed > 2:
        return log.error
    if elapsed > 1:
        return log.warning
    return log.debug


def timer(func):

    actual_func = func
    if isinstance(func, staticmethod):
        actual_func = func.__func__
    elif isinstance(func, classmethod):
        actual_func = func.__func__

    filename = os.path.basename(actual_func.__code__.co_filename)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed > 0.1:
            _get_logger(elapsed)(
                f"⌛️ [{filename}: {func.__name__}] {elapsed * 1000:.0f}ms"
            )
        return result

    return wrapper
