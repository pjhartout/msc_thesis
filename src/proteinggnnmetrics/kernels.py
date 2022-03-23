# -*- coding: utf-8 -*-

"""kernels.py

Kernels

"""

import os
from abc import ABCMeta
from itertools import combinations, combinations_with_replacement, product
from typing import Any, Iterable, List, Union

import networkx as nx
import numpy as np
import pandas as pd
from grakel import WeisfeilerLehman, graph_from_networkx
from gtda.diagrams import PairwiseDistance
from gtda.utils import check_diagrams
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

from .utils.functions import (
    chunks,
    distribute_function,
    flatten_lists,
    generate_random_strings,
    pad_diagrams,
)

default_eigvalue_precision = float("-1e-5")


def positive_eig(K):
    """Assert true if the calculated kernel matrix is valid."""
    min_eig = np.real(np.min(np.linalg.eig(K)[0]))
    np.testing.assert_array_less(default_eigvalue_precision, min_eig)


def distance2similarity(K):
    """
    Convert distance matrix to similarity matrix using a strictly
    monotone decreasing function.
    """
    K = 1 / (1 + K)
    return K


def networkx2grakel(X: Iterable) -> Iterable:
    Xt = list(graph_from_networkx(X, node_labels_tag="residue"))
    return Xt


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


class WeisfeilerLehmanGrakel(Kernel):
    def __init__(
        self, n_jobs: int = 4, n_iter: int = 3, node_label: str = "residue",
    ):
        self.n_iter = n_iter
        self.node_label = node_label
        self.n_jobs = n_jobs

    def compute_gram_matrix(self, X: Any, Y: Any = None) -> Any:
        wl_kernel_grakel = WeisfeilerLehman(
            n_jobs=self.n_jobs, n_iter=self.n_iter
        )
        if Y is None:
            Y = X
        X = networkx2grakel(X)
        Y = networkx2grakel(Y)
        KXY_grakel = wl_kernel_grakel.fit(X).transform(Y).T
        return KXY_grakel


class WassersteinKernel(Kernel):
    def __init__(self, n_jobs: Union[int, None], order: float):
        self.n_jobs = n_jobs
        self.order = order

    def parallel_wasserstein_distance(self, lst: Iterable,) -> Iterable:
        """Computes the pairwise wasserstein distance of elements in lst.

        Args:
            lst (Iterable): Iterable to compute the inner product of.

        Returns:
            Iterable: computed inner products.
        """
        res = list()
        for x in lst:
            res.append(
                {
                    list(x.keys())[0]: [
                        list(x.values())[0][0],
                        self.wasserstein_dist(list(x.values())[0][1]),
                    ]
                }
            )
        return res

    def wasserstein_dist(self, X: Any) -> float:
        pd = PairwiseDistance(
            metric="wasserstein", order=self.order, n_jobs=1,
        )
        # Pad points with duplicate features to allow for concatenation
        X = pad_diagrams(X)
        K = pd.fit_transform(X)[1, 0]
        return K

    def compute_gram_matrix(self, X: List, Y: List = None) -> Any:
        """Apply kernel"""
        if Y is None or X == Y:
            pd = PairwiseDistance(
                metric="wasserstein", order=self.order, n_jobs=self.n_jobs,
            )
            K = pd.fit_transform(X)
        else:
            iters_data = list(list(product(X, Y)))
            iters_idx = list(product(range(len(X)), range(len(Y))))

            keys = generate_random_strings(10, len(flatten_lists(iters_data)))
            iters = [
                {key: [idx, data]}
                for key, idx, data in zip(keys, iters_idx, iters_data)
            ]
            if self.n_jobs is not None:
                iters = list(chunks(iters, self.n_jobs,))
                matrix_elems = flatten_lists(
                    distribute_function(
                        self.parallel_wasserstein_distance,
                        iters,
                        self.n_jobs,
                        tqdm_label="Compute dot products",
                    )
                )
            K = np.zeros((len(X), len(Y)), dtype=float)
            for elem in matrix_elems:
                coords = list(elem.values())[0][0]
                val = list(elem.values())[0][1]
                K[coords[0], coords[1]] = val

            if X == Y:
                # mirror the matrix along diagonal
                K = np.triu(K) + np.triu(K, 1).T

        K = distance2similarity(K)
        return K
