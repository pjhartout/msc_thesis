# -*- coding: utf-8 -*-

"""metrics.py

Metrics

"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances


def _persistence_fisher_distance(D1, D2, kernel_approx=None, bandwidth=1.0):
    projection = (1.0 / 2) * np.ones((2, 2))
    diagonal_projections1 = np.matmul(D1, projection)
    diagonal_projections2 = np.matmul(D2, projection)
    if kernel_approx is not None:
        approx1 = kernel_approx.transform(D1)
        approx_diagonal1 = kernel_approx.transform(diagonal_projections1)
        approx2 = kernel_approx.transform(D2)
        approx_diagonal2 = kernel_approx.transform(diagonal_projections2)
        Z = np.concatenate(
            [approx1, approx_diagonal1, approx2, approx_diagonal2], axis=0
        )
        U, V = (
            np.sum(
                np.concatenate([approx1, approx_diagonal2], axis=0), axis=0
            ),
            np.sum(
                np.concatenate([approx2, approx_diagonal1], axis=0), axis=0
            ),
        )
        vectori, vectorj = np.abs(np.matmul(Z, U.T)), np.abs(np.matmul(Z, V.T))
        vectori_sum, vectorj_sum = np.sum(vectori), np.sum(vectorj)
        if vectori_sum != 0:
            vectori = vectori / vectori_sum
        if vectorj_sum != 0:
            vectorj = vectorj / vectorj_sum
        return np.arccos(min(np.dot(np.sqrt(vectori), np.sqrt(vectorj)), 1.0))
    else:
        Z = np.concatenate(
            [D1, diagonal_projections1, D2, diagonal_projections2], axis=0
        )
        U, V = (
            np.concatenate([D1, diagonal_projections2], axis=0),
            np.concatenate([D2, diagonal_projections1], axis=0),
        )
        vectori = np.sum(
            np.exp(
                -np.square(pairwise_distances(Z, U))
                / (2 * np.square(bandwidth))
            )
            / (bandwidth * np.sqrt(2 * np.pi)),
            axis=1,
        )
        vectorj = np.sum(
            np.exp(
                -np.square(pairwise_distances(Z, V))
                / (2 * np.square(bandwidth))
            )
            / (bandwidth * np.sqrt(2 * np.pi)),
            axis=1,
        )
        vectori_sum, vectorj_sum = np.sum(vectori), np.sum(vectorj)
        if vectori_sum != 0:
            vectori = vectori / vectori_sum
        if vectorj_sum != 0:
            vectorj = vectorj / vectorj_sum
        return np.arccos(min(np.dot(np.sqrt(vectori), np.sqrt(vectorj)), 1.0))


def pairwise_persistence_diagram_distances(X, Y=None, n_jobs=None, **kwargs):
    XX = np.reshape(np.arange(len(X)), [-1, 1])
    YY = (
        None if Y is None or Y is X else np.reshape(np.arange(len(Y)), [-1, 1])
    )

    return _pairwise(
        pairwise_distances,
        True,
        XX,
        YY,
        metric=_sklearn_wrapper(_persistence_fisher_distance, X, Y, **kwargs),
        n_jobs=n_jobs,
    )


def _sklearn_wrapper(metric, X, Y, **kwargs):
    """
    This function is a wrapper for any metric between two persistence diagrams that takes two numpy arrays of shapes (nx2) and (mx2) as arguments.
    """
    if Y is None:

        def flat_metric(a, b):
            return metric(X[int(a[0])], X[int(b[0])], **kwargs)

    else:

        def flat_metric(a, b):
            return metric(X[int(a[0])], Y[int(b[0])], **kwargs)

    return flat_metric


def pairwise_persistence_diagram_kernels(X, Y=None, n_jobs=None, **kwargs):
    XX = np.reshape(np.arange(len(X)), [-1, 1])
    YY = (
        None if Y is None or Y is X else np.reshape(np.arange(len(Y)), [-1, 1])
    )
    return np.exp(
        -pairwise_persistence_diagram_distances(
            X,
            Y,
            kernel_approx=kwargs["kernel_approx"],
            bandwidth=kwargs["bandwidth"],
            n_jobs=n_jobs,
        )
        / kwargs["bandwidth_fisher"]
    )


def _pairwise(fallback, skipdiag, X, Y, metric, n_jobs):
    if Y is not None:
        return fallback(X, Y, metric=metric, n_jobs=n_jobs)
    triu = np.triu_indices(len(X), k=skipdiag)
    tril = (triu[1], triu[0])
    par = Parallel(n_jobs=n_jobs, prefer="threads")
    d = par(
        delayed(metric)([triu[0][i]], [triu[1][i]])
        for i in range(len(triu[0]))
    )
    m = np.empty((len(X), len(X)))
    m[triu] = d
    m[tril] = d
    if skipdiag:
        np.fill_diagonal(m, 0)
    return m
