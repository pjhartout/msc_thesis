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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances, pairwise_kernels
from sklearn.metrics.pairwise import linear_kernel

from .utils.metrics import (
    _persistence_fisher_distance,
    pairwise_persistence_diagram_kernels,
)
from .utils.preprocessing import (
    Padding,
    filter_dimension,
    remove_giotto_pd_padding,
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


# The classes below are taken and modified parts of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.


class PersistenceFisherKernel(BaseEstimator, TransformerMixin, Kernel):
    def __init__(
        self,
        bandwidth_fisher=1.0,
        bandwidth=1.0,
        kernel_approx=None,
        n_jobs=None,
    ):
        self.bandwidth = bandwidth
        self.bandwidth_fisher, self.kernel_approx = (
            bandwidth_fisher,
            kernel_approx,
        )
        self.n_jobs = n_jobs

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

    def compute_gram_matrix(self, X: Any, Y: Any = None) -> Any:
        # Remove points where birth = death
        # Loop over homology dimensions

        X = remove_giotto_pd_padding(X)
        if Y is not None:
            Y = remove_giotto_pd_padding(Y)

        Ks = list()
        for homology_dimension in X[0]["dim"].unique():
            # Get the diagrams for the homology dimension
            X_diag = filter_dimension(X, homology_dimension)
            # X_diag = Padding(use=True).fit_transform(X_diag)
            if Y is not None:
                Y_diag = filter_dimension(Y, homology_dimension)
                # Y_diag = Padding(use=True).fit_transform(Y_diag)

            if Y is not None:
                Ks.append(self.fit(X_diag).transform(Y_diag))
            else:
                Ks.append(self.fit_transform(X_diag))
        # We take the average of the kernel matrices in each homology dimension
        return np.average(np.array(Ks), axis=0)
