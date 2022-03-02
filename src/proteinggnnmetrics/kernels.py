# -*- coding: utf-8 -*-

"""kernels.py

Kernels

"""

import os
from abc import ABCMeta
from itertools import product
from typing import Any, Callable, Iterable, List, Union

import networkx as nx
import numpy as np
from grakel import graph_from_networkx, kernels
from grakel.kernels import VertexHistogram, WeisfeilerLehman
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

from .utils.functions import distribute_function, networkx2grakel


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


class WeisfeilerLehmanKernel(Kernel):
    """Weisfeiler-Lehmann kernel"""

    def __init__(
        self,
        n_iter: int,
        base_graph_kernel: Any,
        normalize: bool,
        n_jobs: int,
        pre_computed_hash,
    ):
        self.n_iter = n_iter
        self.base_graph_kernel = base_graph_kernel
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.pre_computed_hash = pre_computed_hash

    def compute_naive_kernel_matrix(
        self, X: Any, Y: Any, fit: bool
    ) -> np.ndarray:
        gk = WeisfeilerLehman(
            n_iter=self.n_iter,
            base_graph_kernel=self.base_graph_kernel,
            normalize=self.normalize,
            n_jobs=self.n_jobs,
        )

        X = networkx2grakel(X)
        if Y is not None:
            Y = graph_from_networkx(X)
            if fit:
                return gk.fit_transform(X, Y)
            else:
                return gk.transform(X, Y)
        else:
            if fit:
                return gk.fit_transform(X)
            else:
                return gk.transform(X)

    def compute_prehashed_kernel_matrix(
        self, X: Iterable, Y: Union[Iterable, None]
    ) -> List:
        def dot_product(dicts):
            running_sum = 0
            for key in set(dicts[0].keys()).intersection(dicts[1].keys()):
                running_sum += dicts[0][key] * dicts[1][key]
            return running_sum

        if Y == None:
            Y = X

        iters = list(product(X, Y))
        res = list()
        for x in tqdm(
            iters, desc="Dot product of elements in matrix", total=len(iters)
        ):
            res.append(dot_product(x))

        return res

    def fit_transform(self, X: Any, Y: Any = None) -> np.ndarray:
        if self.pre_computed_hash:
            return self.compute_prehashed_kernel_matrix(X, Y)
        else:
            return self.compute_naive_kernel_matrix(X, Y, fit=True)

    def transform(self, X: Any, Y: Any = None) -> np.ndarray:
        if self.pre_computed_hash:
            return self.compute_prehashed_kernel_matrix(X, Y)
        else:
            return self.compute_naive_kernel_matrix(X, Y, fit=False)


class LinearKernel(Kernel):
    def __init__(
        self, dense_output: bool = False,
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
