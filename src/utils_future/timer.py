import functools
import os
import time
from typing import Any

from utils_future.Log import Log

log = Log("timer")


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
        end = time.perf_counter()
        elapsed = end - start

        time_str = f"{elapsed * 1000:.2f}ms"

        log.debug(f"⌛️ [{filename}{func.__name__}] {time_str}")
        return result

    return wrapper
