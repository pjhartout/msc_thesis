#!/usr/bin/env python
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

from proteinggnnmetrics.constants import N_JOBS
from proteinggnnmetrics.paths import (
    HUMAN_PROTEOME,
    HUMAN_PROTEOME_CA_CONTACT_MAP,
)
from proteinggnnmetrics.utils import write_matrix


class GraphConstruction(metaclass=ABCMeta):
    def __init__(self, random_state):
        self.random_state = random_state

    def transform(self, graph):
        """Conversion function."""


class ContactMap(GraphConstruction):
    """Extract contact map from pdb file (fully connected weighted graph)"""

    def __init__(
        self,
        granularity: str,
        metric="euclidean",
        p=2,
        metric_params=None,
        n_jobs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.granularity = granularity

    def transform(self, fname, n_jobs):
        """Extract contact map from contents of fname"""
        file_path = Path(fname)
        parser = PDBParser()

        structure = parser.get_structure(
            file_path.stem, HUMAN_PROTEOME / file_path
        )

        residues = [
            r for r in structure.get_residues() if r.get_id()[0] == " "
        ]

        coordinates = list()
        for x in range(len(residues)):
            coordinates.append(residues[x][self.granularity].get_coord())
        coordinates = np.vstack(coordinates)
        return pairwise_distances(
            coordinates, metric="euclidean", n_jobs=n_jobs
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


if __name__ == "__main__":
    main()
