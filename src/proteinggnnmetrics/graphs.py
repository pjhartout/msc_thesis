# -*- coding: utf-8 -*-

"""graph_construction.py

Handles graph extraction from .pdb files

TODO: docstrings

"""

import os
import random
from abc import ABCMeta
from functools import partial
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from .constants import N_JOBS
from .utils.utils import tqdm_joblib
from .utils.validation import check_graph


class GraphConstruction(metaclass=ABCMeta):
    def __init__(self):
        pass

    def transform(self, graph):
        """Conversion function."""


class ContactMap(GraphConstruction):
    """Extract contact map from pdb file (fully connected weighted graph)"""

    def __init__(
        self,
        n_jobs=N_JOBS,
        metric="euclidean",
        p=2,
        metric_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
        self.metric = metric

    def transform(self, X: List):
        """Extract contact map from contents of fname"""

        def compute_contact_map(X):
            return pairwise_distances(
                X, metric=self.metric, n_jobs=self.n_jobs
            )

        pairwise_dist_list = list()
        with tqdm_joblib(
            tqdm(desc=f"Extracting contact map", total=len(X),)
        ) as progressbar:
            Xt = Parallel(n_jobs=N_JOBS)(
                delayed(compute_contact_map)(sample) for sample in X
            )
        return Xt


class KNNGraph(GraphConstruction):
    """Extract KNN graph"""

    def __init__(
        self,
        n_neighbors: str,
        n_jobs=N_JOBS,
        mode="connectivity",
        metric="euclidean",
        p=2,
        metric_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def transform(self, X: np.ndarray):
        """Extract contact map from contents of fname"""
        X = check_graph(X)

        def knn_graph_func(X):
            return kneighbors_graph(
                X,
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                p=self.p,
                metric_params=self.metric_params,
                mode=self.mode,
                include_self=False,
            )

        with tqdm_joblib(
            tqdm(desc=f"Extracting KNN graph from contact map", total=len(X),)
        ) as progressbar:
            Xt = Parallel(n_jobs=self.n_jobs)(
                delayed(knn_graph_func)(sample) for sample in X
            )
        return Xt


class EpsilonGraph(GraphConstruction):
    """Extract epsilon graph"""

    def __init__(
        self, epsilon: float = 3.0, n_jobs: int = N_JOBS, **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_jobs = n_jobs

    def transform(self, X: np.ndarray):
        """Extract contact map from contents of fname"""
        X = check_graph(X)

        def epsilon_graph_func_(X):
            return np.where(X < self.epsilon, 1, 0)

        with tqdm_joblib(
            tqdm(desc=f"Extracting epsilon graph", total=len(X),)
        ) as progressbar:
            Xt = Parallel(n_jobs=self.n_jobs)(
                delayed(epsilon_graph_func_)(sample) for sample in X
            )
        return Xt
