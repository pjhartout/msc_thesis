# -*- coding: utf-8 -*-

"""kernels.py

Kernels

"""

import os
from abc import ABCMeta
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm


class Kernel(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self):
        pass

    def compute_gram_matrix(self, X: Any, Y: Any = None) -> Any:
        """Apply transformation to apply kernel to X"""
        pass

class LinearKernel(Kernel):
    def __init__(
        self, dense_output: bool = False,
    ):
        self.dense_output = dense_output

    def compute_gram_matrix(self, X: Any, Y: Any = None) -> Any:
        if Y is None:
            K = linear_kernel(X, X, dense_output=self.dense_output)
        else:
            K = linear_kernel(X, Y, dense_output=self.dense_output)
        return K
