# -*- coding: utf-8 -*-
import functools
import time
import tracemalloc
from typing import Callable


def timeit(func: Callable) -> Callable:
    """Timeit decorator

    Args:
        func (Callable): function to time

    Returns:
        Callable: function used to time
    """

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"Callable: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure


def measure_memory(func: Callable, *args, **kwargs):
    """Decorator used to measure memory footprint
    Partly from: https://gist.github.com/davidbegin/d24e25847a952441b62583d973b6c62e

    Args:
        func (Callable): function to measure memory footprint of.
    """

    @functools.wraps(func)
    def memory_closure(*args, **kwargs):
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        print(
            f"\n\033[37mFunction Name       :\033[35;1m {func.__name__}\033[0m"
        )
        print(
            f"\033[37mCurrent memory usage:\033[36m {current / 10**6}MB\033[0m"
        )
        print(f"\033[37mPeak                :\033[36m {peak / 10**6}MB\033[0m")
        tracemalloc.stop()
        return result

    return memory_closure
