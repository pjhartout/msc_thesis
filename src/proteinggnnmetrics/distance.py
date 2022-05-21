# -*- coding: utf-8 -*-

"""distance.py

Distance functions.


TODO: docs, tests, citations.
"""

from abc import ABCMeta
from ctypes import Union
from math import sqrt
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import numpy.typing as npt
from grakel import WeisfeilerLehman
from gtda.diagrams import PairwiseDistance
from scipy.spatial.distance import minkowski
from scipy.stats import pearsonr, spearmanr

from .kernels import Kernel
from .utils.functions import networkx2grakel, positive_eig
from .utils.validation import check_dist


class MaximumMeanDiscrepancy:
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
        self,
        kernel: Kernel = None,
        biased: bool = False,
        squared: bool = True,
        from_sum: bool = False,
        verbose: bool = False,
    ):
        self.kernel = kernel
        self.biased = biased
        self.squared = squared
        self.from_sum = from_sum
        self.verbose = verbose

    def compute(self, *args) -> float:
        if len(args) == 2:
            Xt = check_dist(args[0])
            Yt = check_dist(args[1])

        elif len(args) == 3:
            K_XX = args[0]
            K_YY = args[1]
            K_XY = args[2]

        else:
            raise ValueError(
                "MaximumMeanDiscrepancy.compute() takes either 2 arguments "
                "(two distributions) or 3 arguments (precomputed kernel matrices)."
            )

        if len(args) <= 2:
            K_XX = self.kernel.compute_gram_matrix(Xt)
            K_YY = self.kernel.compute_gram_matrix(Yt)
            K_XY = self.kernel.compute_gram_matrix(Xt, Yt)

        # Following the notation of the paper
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        if self.verbose:
            # Print min_eig for K_XX, K_YY and K_XY
            print(f"Kernel: {self.kernel.__class__.__name__}")
            print(f"K_XX min_eig: {positive_eig(K_XX)}")
            print(f"K_YY min_eig: {positive_eig(K_YY)}")

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

    def compute_from_sums(
        self, k_XX, k_XY, k_YY, m, n,
    ):

        if self.biased:
            mmd = (
                1 / (m ** 2) * k_XX + 1 / (n ** 2) * k_YY - 2 / (m * n) * k_XY
            )

        else:
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


class MinkowskyDistance:
    """Implements maximum mean discrepancy"""

    def __init__(self, p=2):
        # Default set to 2 to recover Euclidean distance.
        self.p = p

    def compute(self, X: np.ndarray, Y: np.ndarray) -> float:
        return minkowski(X.flatten(), Y.flatten(), self.p)


class SpearmanCorrelation:
    """Spearman correlation coefficient"""

    def __init__(self, p_value: bool = False):
        self.p_value = p_value

    def compute(self, X: np.ndarray, Y: np.ndarray) -> float:
        correlation, p = spearmanr(X.flatten(), Y.flatten())
        if self.p_value:
            return correlation, p
        else:
            return correlation


class PearsonCorrelation:
    """Spearman correlation coefficient"""

    def __init__(self, p_value: bool = False):
        self.p_value = p_value  # p is two-sided here. No other choice.

    def compute(self, X: np.ndarray, Y: np.ndarray) -> float:
        correlation, p = pearsonr(X.flatten(), Y.flatten(),)
        if self.p_value:
            return correlation, p
        else:
            return correlation
