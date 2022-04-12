# -*- coding: utf-8 -*-

"""debug.py

This file provides some functions useful for debugging.


"""

import functools
import time
import tracemalloc
from typing import Callable, List

import numpy as np

from ..protein import Protein
from .colors import bcolors


class SamplePoints:
    """SamplePoints class

    This class is used to sample points from a given point cloud.
    This accelerates processing during debugging

    """

    def __init__(self, n_points: float):
        """__init__ method

        Args:
            frac (int): number of points to sample
        """
        self.n_points = n_points

    def fit(self, X, y=None):
        """fit method"""
        pass

    def transform(self, X):
        """transform method"""
        return X

    def fit_transform(self, X: List[Protein], y=None):
        """fit_transform method"""
        for protein in X:
            protein.coordinates = protein.coordinates[: self.n_points]
        return X


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
        print(
            "Function Name       :"
            + bcolors.OKGREEN
            + f"{func.__name__}"
            + bcolors.ENDC
        )
        print(
            "Time                :"
            + bcolors.OKGREEN
            + f"{time_elapsed} seconds"
            + bcolors.ENDC
        )
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
            "Function Name       :"
            + bcolors.OKGREEN
            + f"{func.__name__}"
            + bcolors.ENDC
        )
        print(
            "Current memory usage:"
            + bcolors.OKGREEN
            + f"{current / 10**6}MB"
            + bcolors.ENDC
        )
        print(
            f"Peak                :"
            + bcolors.OKGREEN
            + f"{peak / 10**6}MB"
            + bcolors.ENDC
        )
        tracemalloc.stop()
        return result

    return memory_closure
