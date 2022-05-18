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
from grakel import WeisfeilerLehman, graph_from_networkx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances, pairwise_kernels
from sklearn.metrics.pairwise import linear_kernel

from .utils.functions import distribute_function, networkx2grakel
from .utils.metrics import (
    _persistence_fisher_distance,
    pairwise_persistence_diagram_kernels,
)
from .utils.preprocessing import filter_dimension, remove_giotto_pd_padding

default_eigvalue_precision = float("-1e-5")


class Kernel(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self, n_jobs: int, verbose: bool = False):
        self.n_jobs = n_jobs
        self.verbose = verbose

    def compute_matrix(self, X: Any, Y: Any = None) -> Any:
        """Apply transformation to apply kernel to X"""
        pass


class LinearKernel(Kernel):
    def __init__(
        self, dense_output: bool = False, normalize: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.dense_output = dense_output
        self.normalize = normalize

    def compute_matrix(
        self, X: np.ndarray, Y: Union[np.ndarray, None] = None
    ) -> Any:
        if Y is not None:
            if self.normalize:
                if X.shape == Y.shape:
                    return np.dot(X, Y.T) / np.sqrt(
                        np.dot(X, X.T) * np.dot(Y, Y.T)
                    )
                else:
                    raise ValueError(
                        "X and Y must have the same shape to normalize"
                    )
            return np.dot(X, Y.T)
        else:
            K_XX = np.dot(X, X.T)
            if self.normalize:
                return K_XX / np.sqrt(K_XX * K_XX)
            return K_XX


class GaussianKernel(Kernel):
    def __init__(
        self,
        sigma: float,
        pre_computed_difference: bool,
        return_product: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.pre_computed_difference = pre_computed_difference
        self.return_product = return_product

    def compute_matrix(
        self, X: np.ndarray, Y: np.ndarray = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:

        if self.pre_computed_difference:
            P = np.dot(X - Y, X - Y)
        if self.return_product:
            return np.exp(-self.sigma * P), P
        else:
            return np.exp(-self.sigma * P)


class WeisfeilerLehmanGrakel(Kernel):
    def __init__(self, n_iter: int = 3, node_label: str = "residue", **kwargs):
        super().__init__(**kwargs)
        self.n_iter = n_iter
        self.node_label = node_label

    def compute_matrix(self, X: Any, Y: Any = None) -> Any:
        wl_kernel_grakel = WeisfeilerLehman(
            n_jobs=self.n_jobs, n_iter=self.n_iter
        )
        if Y is None:
            Y = X
        X = networkx2grakel(X)
        Y = networkx2grakel(Y)
        KXY_grakel = wl_kernel_grakel.fit(X).transform(Y).T
        return KXY_grakel


# The classes below are taken and modified parts of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.


class PersistenceFisherKernel(BaseEstimator, TransformerMixin, Kernel):
    def __init__(
        self, bandwidth_fisher=1.0, bandwidth=1.0, kernel_approx=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.bandwidth_fisher, self.kernel_approx = (
            bandwidth_fisher,
            kernel_approx,
        )

    def fit(self, X, y=None):
        self.diagrams_ = X
        return self

    def transform(self, X):
        return pairwise_persistence_diagram_kernels(
            X,
            self.diagrams_,
            kernel="persistence_fisher",
            bandwidth=self.bandwidth,
            bandwidth_fisher=self.bandwidth_fisher,
            kernel_approx=self.kernel_approx,
            n_jobs=self.n_jobs,
        )

    def __call__(self, diag1, diag2):
        return (
            np.exp(
                -_persistence_fisher_distance(
                    diag1,
                    diag2,
                    bandwidth=self.bandwidth,
                    kernel_approx=self.kernel_approx,
                )
            )
            / self.bandwidth_fisher
        )

    def compute_matrix(self, X: Any, Y: Any = None) -> Any:
        # Remove points where birth = death
        # Loop over homology dimensions

        # X = remove_giotto_pd_padding(X)
        # if Y is not None:
        #     Y = remove_giotto_pd_padding(Y)

        def compute_kernel_in_homology_dimension(homology_dimension):
            # Get the diagrams for the homology dimension
            X_diag = filter_dimension(X, homology_dimension)
            # X_diag = Padding(use=True).fit_transform(X_diag)
            if Y is not None:
                Y_diag = filter_dimension(Y, homology_dimension)
                # Y_diag = Padding(use=True).fit_transform(Y_diag)

            if Y is not None:
                return self.fit(X_diag).transform(Y_diag)
            else:
                return self.fit_transform(X_diag)

        Ks = distribute_function(
            compute_kernel_in_homology_dimension,
            np.unique(X[0][:, 2]),
            n_jobs=self.n_jobs,
            tqdm_label="Computing Persistence Fisher Kernel",
            show_tqdm=self.verbose,
        )

        # We take the average of the kernel matrices in each homology dimension
        return np.product(np.array(Ks), axis=0)


class KernelComposition:
    def __init__(
        self,
        kernels: List[Kernel],
        composition_rule: Union[str, List[str]],
        kernel2reps: List[int],
    ):
        self.kernels = kernels
        self.composition_rule = composition_rule
        self.kernel2reps = kernel2reps

    def compute_matrix(self, X_reps: List, Y_reps: List = None) -> Any:
        """Because this kernel can operate on multiple representations of the data (graphs, embeddings, graph descriptors, sequence data, etc.), we provide an extra argument to the compute_matrix method. This argument is a list that gives the index of the input vector to use with the kernel.

        Example:
        >>> kernel2reps = [0, 1, 2]
        >>> X_reps = [X_graphs, X_embeddings, X_graph_descriptors]
        >>> kernel = KernelComposition([Kernel1, Kernel2, Kernel3], composition_rule="product")
        >>> kernel.compute_matrix(X_reps, Y_reps, kernel2reps)

        Args:
            X_reps (List): representations of X
            Y_reps (List): representations of Y
            kernel2reps (Dict): index of representations to use

        Raises:
            ValueError: if supplied inputs are incorrect. See error messages.

        Returns:
            Any: composed kernel matrix
        """
        if Y_reps is None:
            Y_reps = X_reps

        if len(X_reps) != len(self.kernel2reps) or len(Y_reps) != len(
            self.kernel2reps
        ):
            raise ValueError(
                "The number of input representations must match the number of kernels."
            )

        # Compute kernel matrices using kernels
        K_list = []
        for idx, kernel in enumerate(self.kernels):
            K_list.append(
                kernel.compute_matrix(
                    X_reps[self.kernel2reps[idx]],
                    Y_reps[self.kernel2reps[idx]],
                )
            )

        # Compute the composition rule
        if isinstance(self.composition_rule, str):
            if self.composition_rule == "product":
                return np.prod(K_list, axis=0)
            elif self.composition_rule == "sum":
                return np.sum(K_list, axis=0)
            else:
                raise ValueError(
                    "The composition rule {} is not supported".format(
                        self.composition_rule
                    )
                )
        elif isinstance(self.composition_rule, list):
            if len(self.composition_rule) != len(K_list) - 1:
                raise ValueError(
                    f"The length of the list of composition rules is incompatible with the number of kernels"
                )
            else:
                K = K_list[0]
                for k, rule in zip(K_list[1:], self.composition_rule):
                    if rule == "sum":
                        K += k
                    elif rule == "product":
                        K *= k
                    else:
                        raise ValueError(
                            "The composition rule {} is not supported".format(
                                rule
                            )
                        )
                return K
