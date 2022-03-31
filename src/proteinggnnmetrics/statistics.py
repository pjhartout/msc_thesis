# -*- coding: utf-8 -*-

"""statistics.py

Implements various statistical testing function

TODO: docs, tests, citations, type hints.
"""


import pickle
from collections import Counter
from pathlib import Path, PosixPath
from typing import Callable, Dict, List

import numpy as np
import scipy

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

    def rank_statistic(self, original_mmd, K_XX, K_YY, K_XY):
        """Compute the p-value of the MMD test from the MMD value"""

        # Compute the p-value
        sampled_mmds = list()
        trial_idx = 0
        while trial_idx < self.t:
            # Sample m elements from upper triangular matrix of the original
            # data.
            full_K = np.zeros(
                (K_XX.shape[0] + K_YY.shape[0], K_XX.shape[0] + K_YY.shape[0])
            )
            full_K[: K_XX.shape[0], : K_XX.shape[0]] = K_XX
            full_K[K_XX.shape[0] :, K_XX.shape[0] :] = K_YY
            full_K[K_XX.shape[0] :, : K_XX.shape[0]] = K_XY.T

            # Sample m elements from the upper triangular matrix of the full
            # data.
            triu_K = np.triu_indices(full_K.shape[0], k=1)  # type: ignore
            sampled_indices_rows = np.random.choice(
                triu_K[1], size=self.m, replace=False
            )
            sampled_indices_cols = np.random.choice(
                triu_K[0], size=self.m, replace=False
            )

            # Compute k_XX, k_YY, k_XY
            sampled_K_XX = full_K[
                sampled_indices_rows[
                    np.logical_and(
                        sampled_indices_rows < K_XX.shape[0],
                        sampled_indices_cols < K_XX.shape[0],
                    )
                ],
                sampled_indices_cols[
                    np.logical_and(
                        sampled_indices_rows < K_XX.shape[0],
                        sampled_indices_cols < K_XX.shape[0],
                    )
                ],
            ]
            sampled_K_YY = full_K[
                sampled_indices_rows[
                    np.logical_and(
                        sampled_indices_rows > K_XX.shape[0],
                        sampled_indices_cols > K_XX.shape[0],
                    )
                ],
                sampled_indices_cols[
                    np.logical_and(
                        sampled_indices_rows > K_XX.shape[0],
                        sampled_indices_cols > K_XX.shape[0],
                    )
                ],
            ]
            sampled_K_XY = full_K[
                sampled_indices_rows[
                    np.logical_and(
                        sampled_indices_rows < K_XX.shape[0],
                        sampled_indices_cols >= K_XX.shape[0],
                    )
                ],
                sampled_indices_cols[
                    np.logical_and(
                        sampled_indices_rows < K_XX.shape[0],
                        sampled_indices_cols >= K_XX.shape[0],
                    )
                ],
            ]

            if len(sampled_K_XX) >= 2 and len(sampled_K_YY) >= 2:
                trial_idx += 1
            else:
                break

            # Compute MMD on sampled data.
            sampled_mmds.append(
                MaximumMeanDiscrepancy(
                    biased=False,
                    squared=True,
                    verbose=False,
                ).compute_from_sums(
                    sampled_K_XX.sum(),
                    sampled_K_YY.sum(),
                    sampled_K_XY.sum(),
                    len(sampled_K_XX),
                    len(sampled_K_YY),
                )
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
        p_value = self.rank_statistic(
            mmd_original,
            K_XX,
            K_YY,
            K_XY,
        )

        return p_value
