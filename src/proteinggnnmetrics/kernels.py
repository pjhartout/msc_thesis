# -*- coding: utf-8 -*-

"""kernels.py

Kernels

"""

import os
from abc import ABCMeta
from itertools import combinations, combinations_with_replacement, product
from typing import Any, Iterable, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from grakel import graph_from_networkx, kernels
from grakel.kernels import VertexHistogram, WeisfeilerLehman
from matplotlib.cbook import flatten
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

from .utils.functions import (
    chunks,
    distribute_function,
    flatten_lists,
    generate_random_strings,
    networkx2grakel,
)
from .utils.validation import check_graphs, check_hash


class Kernel(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self):
        pass

    def fit(self, X: Iterable) -> Iterable:
        """required for sklearn compatibility"""
        return X

    def transform(self, X: Iterable) -> Iterable:
        """required for sklearn compatibility"""
        return X

    def fit_transform(self, X: Any) -> Any:
        """Apply transformation to apply kernel to X"""
        pass


class WeisfeilerLehmanKernel(Kernel):
    """Weisfeiler-Lehmann kernel"""

    def __init__(
        self,
        n_jobs: int,
        n_iter: int = 3,
        normalize: bool = True,
        pre_computed_hash: bool = False,
        base_graph_kernel: Any = None,
        biased: bool = True,
        vectorized: bool = True,
    ):
        self.n_iter = n_iter
        self.base_graph_kernel = base_graph_kernel
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.biased = biased
        self.pre_computed_hash = pre_computed_hash
        self.vectorized: bool = vectorized

    def compute_naive_kernel_matrix(
        self, X: Any, Y: Any, fit: bool
    ) -> np.ndarray:
        X = check_graphs(X)

        wl_gk = WeisfeilerLehman(
            n_iter=self.n_iter,
            base_graph_kernel=self.base_graph_kernel,
            normalize=self.normalize,
            n_jobs=self.n_jobs,
        )

        X = networkx2grakel(X)
        if Y is not None:
            Y = graph_from_networkx(X)
            if fit:
                return wl_gk.fit_transform(X, Y)
            else:
                return wl_gk.transform(X, Y)
        else:
            if fit:
                return wl_gk.fit_transform(X)
            else:
                return wl_gk.transform(X)

    def compute_prehashed_kernel_matrix_unordered(self, X, Y):
        X = check_hash(X)
        Y = check_hash(Y)

        def parallel_dot_product(lst: Iterable) -> Iterable:
            res = list()
            for x in lst:
                res.append(
                    {
                        list(x.keys())[0]: [
                            list(x.values())[0][0],
                            dot_product(list(x.values())[0][1]),
                        ]
                    }
                )
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
        iters_data = list(list(product(X, Y)))
        iters_idx = list(product(range(len(X)), range(len(Y))))
        keys = generate_random_strings(10, len(flatten_lists(iters_data)))
        iters = [
            {key: [idx, data]}
            for key, idx, data in zip(keys, iters_idx, iters_data)
        ]

        matrix_elems = parallel_dot_product(iters)

        K = np.zeros((len(X), len(Y)))
        for elem in matrix_elems:
            coords = list(elem.values())[0][0]
            val = list(elem.values())[0][1]
            K[coords[0], coords[1]] = val
        return K

    def compute_prehashed_kernel_matrix_vectorized(
        self, X: Iterable, Y: Union[Iterable, None]
    ) -> np.ndarray:
        def matrix2df(matrix, column_labels):
            return pd.DataFrame(
                matrix.todense(), columns=column_labels
            ).fillna(0)

        def remove_non_overlapping_vectors(Xt, Yt):
            matched_cols = list(set(Xt.columns).intersection(set(Yt.columns)))
            Xt = Xt[matched_cols]
            Yt = Yt[matched_cols]
            return Xt, Yt

        X = check_hash(X)
        Y = check_hash(Y)

        vectorizer = DictVectorizer(dtype=np.uint8, sparse=True)
        Xt = vectorizer.fit_transform(X)
        column_labels = vectorizer.get_feature_names_out()
        Xt = matrix2df(Xt, column_labels)
        if Y == None or Y == X:
            Yt = Xt
        else:
            Yt = vectorizer.fit_transform(Y)
            column_labels = vectorizer.get_feature_names_out()
            Yt = matrix2df(Yt, column_labels)

        Xt, Yt = remove_non_overlapping_vectors(Xt, Yt)

        Xt, Yt = Xt.values, Yt.values

        return Xt.dot(Yt.T)

    def compute_prehashed_kernel_matrix(self, X, Y):
        if self.vectorized:
            return self.compute_prehashed_kernel_matrix_vectorized(X, Y)
        else:
            return self.compute_prehashed_kernel_matrix_unordered(X, Y)

    def fit(self, X: Iterable) -> Iterable:
        """required for sklearn compatibility"""
        return X

    def transform(self, X: Any, Y: Any = None) -> np.ndarray:
        if self.pre_computed_hash:
            return self.compute_prehashed_kernel_matrix(X, Y)
        else:
            return self.compute_naive_kernel_matrix(X, Y, fit=False)

    def fit_transform(self, X: Any, Y: Any = None) -> np.ndarray:
        if self.pre_computed_hash:
            return self.compute_prehashed_kernel_matrix(X, Y)
        else:
            return self.compute_naive_kernel_matrix(X, Y, fit=True)


class LinearKernel(Kernel):
    def __init__(
        self, dense_output: bool = False,
    ):
        self.dense_output = dense_output

    def fit(self, X: Iterable) -> Iterable:
        """required for sklearn compatibility"""
        return X

    def transform(self, X: Iterable) -> Iterable:
        """required for sklearn compatibility"""
        return X

    def fit_transform(self, X: Any, Y: Any = None) -> Any:
        if Y is None:
            K = linear_kernel(X, X, dense_output=self.dense_output)
        else:
            K = linear_kernel(X, Y, dense_output=self.dense_output)
        return K
