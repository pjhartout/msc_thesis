# -*- coding: utf-8 -*-

"""statistics.py

Implements various statistical testing function

TODO: docs, tests, citations, type hints.
"""


import pickle
from collections import Counter
from pathlib import Path, PosixPath
from typing import Callable, Dict, List
import scipy
import numpy as np

from proteinggnnmetrics.kernels import Kernel

from .distance import MaximumMeanDiscrepancy


class MMDTest:
    """Implements the maximum mean discrepancy test
    Overview:

    """

    def __init__(self, alpha: float, m: int, t: int, kernel: Callable) -> None:
        # Following orignal notation in the paper
        self.alpha = alpha
        self.t = t
        self.m = m
        self.kernel = kernel

    def compute_p_value_from_mmd(self, original_mmd, K_XX, K_YY, K_XY):
        """Compute the p-value of the MMD test from the MMD value"""

        # Compute the p-value
        sampled_mmds = list()
        for i in range(self.t):
            # Sample m elements from upper triangular matrix of the original
            # data.
            samples_x = np.random.choice(
                K_XX.shape[0], size=self.m, replace=False
            )
            samples_y = np.random.choice(
                K_YY.shape[0], size=self.m, replace=False
            )
            sampled_K_XX = K_XX[samples_x, :][:, samples_x]
            sampled_K_YY = K_YY[samples_y, :][:, samples_y]
            sampled_K_XY = K_XY[samples_x, :][:, samples_y]

            # Compute MMD on sampled data.
            sampled_mmds.append(
                MaximumMeanDiscrepancy(
                    self.kernel,
                    biased=False,
                    squared=True,
                    verbose=False,
                ).compute(sampled_K_XX, sampled_K_YY, sampled_K_XY)
            )
        sampled_mmds.append(original_mmd)
        rank = np.where(np.sort(np.array(sampled_mmds)) == original_mmd)[0][0]
        p_value = (self.t + 1 - rank) / (self.t + 1)
        return p_value

    def compute_p_value(self, dist_1, dist_2):
        """Compute the p-value of the MMD test"""
        # Compute MMD on original data.
        K_XX = self.kernel.compute_gram_matrix(dist_1)
        K_YY = self.kernel.compute_gram_matrix(dist_2)
        K_XY = self.kernel.compute_gram_matrix(dist_1, dist_2)

        mmd_original = MaximumMeanDiscrepancy(
            kernel=self.kernel,
            biased=False,
            squared=True,
            verbose=False,
        ).compute(K_XX, K_YY, K_XY)

        # Compute the p-value
        p_value = self.compute_p_value_from_mmd(mmd_original, K_XX, K_YY, K_XY)

        return p_value
