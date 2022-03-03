# -*- coding: utf-8 -*-

"""distance.py

Distance functions.


TODO: docs, tests, citations.
"""

from abc import ABCMeta
from typing import Any, Callable

import numpy as np
from scipy.spatial.distance import minkowski

from proteinggnnmetrics.kernels import Kernel

from .utils.validation import check_dist


class DistanceFunction(metaclass=ABCMeta):
    """Defines distance function"""

    def __init__(self):
        pass

    def evaluate(self, X: Any, Y: Any) -> np.ndarray:
        """Apply evaluation of the two input vectors
        """
        pass


class MaximumMeanDiscrepancy(DistanceFunction):
    """Implements maximum mean discrepancy
    """

    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    def evaluate(self, X: Any, Y: Any) -> np.ndarray:
        Xt = check_dist(X)
        Yt = check_dist(Y)

        # Following the original notation of the paper
        m = len(Xt)
        n = len(Yt)

        K_XX = self.kernel.transform(Xt)
        K_YY = self.kernel.transform(Yt)
        K_XY = self.kernel.transform(Xt, Yt)

        # We could also skip diagonal elements in the calculation above but
        # this is more computationally efficient.

        k_XX = np.sum(K_XX)
        k_YY = np.sum(K_YY)
        k_XY = np.sum(K_XY)

        mmd = 1 / (m ** 2) * k_XX + 1 / (n ** 2) * k_YY - 2 / (m * n) * k_XY

        return mmd


class MinkowskyDistance(DistanceFunction):
    """Implements maximum mean discrepancy"""

    def __init__(self, p=2):
        # Default set to 2 to recover Euclidean distance.
        self.p = p

    def evaluate(self, X: Any, Y: Any) -> np.ndarray:
        d = minkowski(X, Y, self.p)
        return d
