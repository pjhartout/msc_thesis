# -*- coding: utf-8 -*-

"""distance.py

Distance functions.


TODO: docs, tests, citations.
"""

from abc import ABCMeta
from math import sqrt
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from gtda.diagrams import PairwiseDistance
from scipy.spatial.distance import minkowski

from .kernels import Kernel
from .utils.validation import check_dist


class DistanceFunction(metaclass=ABCMeta):
    """Defines distance function"""

    def __init__(self):
        pass

    def fit(self):
        """required for sklearn compatibility"""
        pass

    def transform(self):
        """required for sklearn compatibility"""
        pass

    def fit_transform(self, X: Any, Y: Any) -> float:
        """Apply evaluation of the two input vectors"""
        pass


class MaximumMeanDiscrepancy(DistanceFunction):
    """Implements maximum mean discrepancy
    Depending on the configuration, this class implements the following:
    - Unbiased MMD^2: see Lemma 6 of "A Kernel Two-Sample Test" [1]
    - Biased MMD^2: see Equation 5 of "A Kernel Two-Sample Test" [1]
    - Unbiased MMD
    - Biased MMD

    Importantly, the unbiased MMD^2 estimate can be negative.

    References:
    [1]: Gretton, Arthur, Karsten M. Borgwardt, Malte J. Rasch, Bernhard
    Schölkopf, and Alexander Smola. "A kernel two-sample test." The Journal of
    Machine Learning Research 13, no. 1 (2012): 723-773.
    https://jmlr.csail.mit.edu/papers/v13/gretton12a.html

    """

    def __init__(
        self, kernel: Kernel, biased: bool = False, squared: bool = True,
    ):
        self.kernel = kernel
        self.biased = biased
        self.squared = squared

    def fit(self, proteins: Any) -> Any:
        return proteins

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> float:
        Xt = check_dist(X)
        Yt = check_dist(Y)

        # Following the original notation of the paper
        m = len(Xt)
        n = len(Yt)

        K_XX = self.kernel.fit_transform(Xt)
        K_YY = self.kernel.fit_transform(Yt)
        K_XY = self.kernel.fit_transform(Xt, Yt)

        if self.biased:

            k_XX = np.sum(K_XX)
            k_YY = np.sum(K_YY)
            k_XY = np.sum(K_XY)

            mmd = (
                1 / (m ** 2) * k_XX + 1 / (n ** 2) * k_YY - 2 / (m * n) * k_XY
            )

        else:
            np.fill_diagonal(K_XX, 0)
            np.fill_diagonal(K_YY, 0)

            k_XX = np.sum(K_XX)
            k_YY = np.sum(K_YY)
            k_XY = np.sum(K_XY)

            mmd = (
                1 / (m * (m - 1)) * k_XX
                + 1 / (n * (n - 1)) * k_YY
                - 2 / (m * n) * k_XY
            )

        if self.squared:
            return mmd

        else:

            if mmd < 0:
                mmd = 0
                raise RuntimeWarning("Warning: MMD is negative, set to 0")

            # See: https://twitter.com/karpathy/status/1430316576016793600?s=21
            return sqrt(mmd)


class MinkowskyDistance(DistanceFunction):
    """Implements maximum mean discrepancy"""

    def __init__(self, p=2):
        # Default set to 2 to recover Euclidean distance.
        self.p = p

    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> float:
        d = minkowski(X.flatten(), Y.flatten(), self.p)
        return d


class TopologicalPairwiseDistance:
    def __init__(
        self, metric: str, metric_params: Dict, order: int, n_jobs: int
    ):
        self.metric = metric
        self.metric_params = metric_params
        self.order = order
        self.n_jobs = n_jobs

    def fit_transform(
        self, X: Iterable, Y: Iterable
    ) -> Tuple[np.ndarray, np.ndarray]:
        pw_dist_diag = PairwiseDistance(
            metric=self.metric,
            metric_params=self.metric_params,
            order=self.order,
            n_jobs=self.n_jobs,
        )
        return pw_dist_diag.fit_transform(X), pw_dist_diag.fit_transform(Y)
