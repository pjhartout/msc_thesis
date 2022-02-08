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

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from .constants import N_JOBS
from .utils import tqdm_joblib, write_matrix


class GraphConstruction(metaclass=ABCMeta):
    def __init__(self):
        pass

    def transform(self, graph):
        """Conversion function."""


class ContactMap(GraphConstruction):
    """Extract contact map from pdb file (fully connected weighted graph)"""

    def __init__(
        self, metric="euclidean", p=2, metric_params=None, n_jobs=1, **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
        self.metric = metric

    def transform(self, coords: np.ndarray):
        """Extract contact map from contents of fname"""
        return pairwise_distances(
            coords, metric=self.metric, n_jobs=self.n_jobs
        )


class KNNGraph(GraphConstruction):
    """Extract KNN graph"""

    def __init__(
        self,
        n_neighbors: str,
        mode="connectivity",
        metric="euclidean",
        p=2,
        metric_params=None,
        n_jobs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def transform(self, graph: np.ndarray):
        """Extract contact map from contents of fname"""
        _adjacency_matrix_func = partial(
            kneighbors_graph,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            mode=self.mode,
            include_self=False,
        )
        knn_graph = Parallel(n_jobs=self.n_jobs)(
            delayed(_adjacency_matrix_func)(graph_row) for graph_row in graph
        )
        return knn_graph


class EpsilonGraph(GraphConstruction):
    """Extract epsilon graph"""

    def __init__(
        self, epsilon: float, **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def transform(self, graph: np.ndarray):
        """Extract contact map from contents of fname"""
        epsilon_neighborhood_graph = np.where(graph < self.epsilon, 1, 0)
        return epsilon_neighborhood_graph


class ParallelGraphExtraction(GraphConstruction):
    """
    Parallelizes the graph extraction process to process multiple graphs in
    parallel.
    """

    def __init__(self, n_jobs) -> None:
        self.n_jobs = n_jobs

    def transform(self, samples: np.ndarray, func) -> np.ndarray:
        with tqdm_joblib(
            tqdm(
                desc="Extracting coordinates from pdb files",
                total=len(samples),
            )
        ) as progressbar:
            graphs = Parallel(n_jobs=N_JOBS)(
                delayed(func)(sample) for sample in samples
            )

        return graphs
