# -*- coding: utf-8 -*-

"""statistics.py

Implements various statistical testing function

TODO: docs, tests, citations, type hints.
"""


import itertools
import pickle
from collections import Counter
from pathlib import Path, PosixPath
from random import sample
from typing import Callable, Dict, List

import numpy as np
import scipy

from proteinggnnmetrics.kernels import Kernel

from .distance import MaximumMeanDiscrepancy


class MMDTest:
    """Implements the maximum mean discrepancy test
    Overview:

    """

    def __init__(
        self, alpha: float, m: int, t: int, kernel: Callable, verbose: bool
    ) -> None:
        # Following orignal notation in the paper
        self.alpha = alpha
        self.t = t
        self.m = m
        self.kernel = kernel
        self.verbose = verbose

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
            full_K[: K_XX.shape[0], K_XX.shape[0] :] = K_XY
            full_K[K_XX.shape[0] :, : K_XX.shape[0]] = K_XY.T
            # Sample m elements from the upper triangular matrix of the full
            # data.
            sample_x = np.random.choice(
                range(K_XX.shape[0]), self.m, replace=False
            )
            sample_y = np.random.choice(
                range(K_XX.shape[0]), self.m, replace=False
            )

            idx_x = np.concatenate(
                (
                    np.array(
                        [
                            [el1, el2]
                            for el1, el2 in itertools.product(
                                sample_x, sample_x
                            )
                        ]
                    ),
                    np.array(
                        [
                            [el2, el1]
                            for el1, el2 in itertools.product(
                                sample_x, sample_x
                            )
                            if el1 != el2
                        ]
                    ),
                ),
                axis=0,
            )

            idx_y = np.concatenate(
                (
                    np.array(
                        [
                            [el1 + K_XX.shape[0], el2 + K_XX.shape[0]]
                            for el1, el2 in itertools.product(
                                sample_y, sample_y
                            )
                        ]
                    ),
                    np.array(
                        [
                            [el2 + K_XX.shape[0], el1 + K_XX.shape[0]]
                            for el1, el2 in itertools.product(
                                sample_y, sample_y
                            )
                            if el1 != el2
                        ]
                    ),
                ),
                axis=0,
            )

            idx_xy = np.concatenate(
                (
                    np.array(
                        [
                            [el1, el2 + K_XX.shape[0]]
                            for el1, el2 in itertools.product(
                                sample_x, sample_y
                            )
                        ]
                    ),
                    np.array(
                        [
                            [el2 + K_XX.shape[0], el1]
                            for el1, el2 in itertools.product(
                                sample_x, sample_y
                            )
                        ]
                    ),
                ),
                axis=0,
            )

            sampled_K_XX = full_K[idx_x[:, 0], idx_x[:, 1]]
            sampled_K_YY = full_K[idx_y[:, 0], idx_y[:, 1]]
            sampled_K_XY = full_K[idx_xy[:, 0], idx_xy[:, 1]]

            if len(sampled_K_XX) >= 2 and len(sampled_K_YY) >= 2:
                trial_idx += 1
                if self.verbose:
                    print(
                        "Trial {}/{}".format(trial_idx, self.t),
                        flush=True,
                        end="\r",
                    )
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
        print("\n")
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
