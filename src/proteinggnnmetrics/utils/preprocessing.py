# -*- coding: utf-8 -*-

"""preprocessing.py

Handles some preprocessing

"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def remove_giotto_pd_padding(diags):
    diags_unpadded = list()
    for diag in diags:
        diag = pd.DataFrame(diag, columns=["birth", "death", "dim"])
        diags_unpadded.append(diag[diag["birth"] != diag["death"]])
    return diags_unpadded


def filter_dimension(diags, dim):
    diags_filtered = list()
    for diag in diags:
        diag = pd.DataFrame(diag, columns=["birth", "death", "dim"])
        diags_filtered.append(
            diag[diag["dim"] == dim].drop(labels="dim", axis=1).values
        )
    return diags_filtered


class Padding(BaseEstimator, TransformerMixin):
    def __init__(self, use=False):
        self.use = use

    def fit(self, X, y=None):
        self.max_pts = max([len(diag) for diag in X])
        return self

    def transform(self, X):
        """
        All points are given an additional coordinate indicating if the point
        was added after padding (0) or already present before (1).

        """
        if self.use:
            Xfit, num_diag = [], len(X)
            for diag in X:
                diag_pad = np.pad(
                    diag,
                    ((0, max(0, self.max_pts - diag.shape[0])), (0, 1)),
                    "constant",
                    constant_values=((0, 0), (0, 0)),
                )
                diag_pad[: diag.shape[0], 2] = np.ones(diag.shape[0])
                Xfit.append(diag_pad)
        else:
            Xfit = X
        return Xfit

    def __call__(self, diag):
        return self.fit_transform([diag])[0]
