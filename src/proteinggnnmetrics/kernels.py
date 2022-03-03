# -*- coding: utf-8 -*-

"""kernels.py

Kernels

"""

import itertools
import os
from abc import ABCMeta
from typing import Any, Iterable, List, Tuple, Union

import networkx as nx
import numpy as np
from grakel import graph_from_networkx, kernels
from grakel.kernels import VertexHistogram, WeisfeilerLehman
from sklearn.metrics.pairwise import linear_kernel
from sklearn.utils.validation import check_array
from tqdm import tqdm

from .utils.functions import (
    chunks,
    distribute_function,
    flatten_lists,
    networkx2grakel,
)
from .utils.validation import check_graphs, check_hash


class Kernel(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self):
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
        X = check_graphs(X)
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
    ) -> Iterable:
        X = check_hash(X)
        if Y is not None:
            Y = check_hash(Y)

        def parallel_dot_product(lst: Iterable) -> Iterable:
            res = list()
            for x in lst:
                res.append(dot_product(x))
            return res

        def dot_product(dicts: Tuple) -> int:
            running_sum = 0
            # 0 * x = 0 so we only need to iterate over common keys
            for key in set(dicts[0].keys()).intersection(dicts[1].keys()):
                running_sum += dicts[0][key] * dicts[1][key]
            return running_sum

        if Y == None:
            Y = X

        # It's faster to process n_jobs lists than to have one list and
        # dispatch one item at a time.
        iters = list(chunks(list(itertools.product(X, Y)), self.n_jobs))

        return flatten_lists(
            distribute_function(
                parallel_dot_product, iters, n_jobs=self.n_jobs,
            )
        )

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

    def transform(self, X: Any, Y: Any = None) -> Any:
        X = check_array(X)

        if Y == None:
            K = linear_kernel(X, X, dense_output=self.dense_output)
        else:
            K = linear_kernel(X, Y, dense_output=self.dense_output)
        return K
