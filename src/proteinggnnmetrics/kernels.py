# -*- coding: utf-8 -*-

"""kernels.py

Kernels

"""

import os
from abc import ABCMeta
from typing import Any, Callable
from xmlrpc.client import Boolean

import networkx as nx
import numpy as np
from grakel import graph_from_networkx, kernels
from grakel.kernels import VertexHistogram, WeisfeilerLehman
from sklearn.metrics.pairwise import linear_kernel

from proteinggnnmetrics.constants import N_JOBS

from .utils.utils import distribute_function, networkx2grakel


class Kernel(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self):
        pass

    def fit_transform(self, X: Any) -> Any:
        """Apply fitting and transformation to apply kernel to X"""
        pass

    def transform(self, X: Any) -> Any:
        """Apply transformation to apply kernel to X"""
        pass


class WLKernel(Kernel):
    """Weisfeiler-Lehmann kernel"""

    def __init__(
        self,
        n_iter: int = 4,
        base_graph_kernel: kernels = VertexHistogram,
        normalize: bool = True,
    ):
        self.n_iter = n_iter
        self.base_graph_kernel = base_graph_kernel
        self.normalize = normalize
        super().__init__()

    def compute_kernel_matrix(self, X: Any, Y: Any,) -> np.ndarray:

        gk = WeisfeilerLehman(
            n_iter=self.n_iter,
            base_graph_kernel=self.base_graph_kernel,
            normalize=self.normalize,
        )

        X = networkx2grakel(X)
        if Y is not None:
            Y = graph_from_networkx(X)
            return gk.fit_transform(X, Y)
        else:
            return gk.fit_transform(X)

    def fit_transform(self, X: Any, Y: Any = None) -> np.ndarray:
        return self.compute_kernel_matrix(X, Y)

    def transform(self, X: Any, Y: Any = None) -> np.ndarray:
        return self.compute_kernel_matrix(X, Y)


class PreComputedWLKernel(Kernel):
    def __init__(self,):
        pass

    def compute_kernel_matrix(self, X: Any, Y: Any) -> np.ndarray:
        def product(dicts):
            running_sum = 0
            for key in dicts[0]:
                running_sum += dicts[0][key] * dicts[1][key]
            return running_sum

        if Y == None:
            Y = X

        # Parallelize
        res = distribute_function(
            product,
            zip(X, Y),
            "pre-computed_product",
            N_JOBS,
            max([len(X), len(Y)]),
        )
        return res

    def fit_transform(self, X: Any, Y: Any = None) -> np.ndarray:
        return self.compute_kernel_matrix(X, Y)

    def transform(self, X: Any, Y: Any = None) -> np.ndarray:
        return self.compute_kernel_matrix(X, Y)


class LinearKernel(Kernel):
    def __init__(
        self, dense_output: Boolean = False,
    ):
        self.dense_output = dense_output

    def compute_kernel_matrix(self, X: Any, Y) -> Any:
        if Y == None:
            K = linear_kernel(X, X, dense_output=self.dense_output)
        else:
            K = linear_kernel(X, Y, dense_output=self.dense_output)
        return K

    def transform(self, X: Any, Y: Any = None) -> Any:
        return self.compute_kernel_matrix(X, Y)

    def fit_transform(self, X: Any, Y: Any = None) -> Any:
        return self.compute_kernel_matrix(X, Y)


class KernelMatrix:
    """Kernel matrix"""

    def __init__(self) -> None:
        pass
