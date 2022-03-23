# -*- coding: utf-8 -*-

"""graphs.py

Handles graph extraction from a set of atom coordinates.

TODO: check docstrings, citations
"""

from abc import ABCMeta
from typing import Dict, List, Union

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from .protein import Protein
from .utils.functions import distribute_function, tqdm_joblib
from .utils.validation import check_graphs


class GraphConstruction(metaclass=ABCMeta):
    """Defines skeleton of graph construction classes"""

    def __init__(self):
        pass

    def fit(self, proteins: List[Protein]) -> List[Protein]:
        """required for sklearn compatibility"""
        return proteins

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        """required for sklearn compatibility"""
        return proteins

    def fit_transform(self, proteins: List[Protein]) -> List[Protein]:
        """Apply transformation

        Args:
            graphs (Any): list or array of samples

        Returns:
            np.ndarray: array of graphs
        """
        return proteins


class ContactMap(GraphConstruction):
    """Extract contact map from pdb file (fully connected weighted graph)

    Args:
        GraphConstruction (type): class constructor
    """

    def __init__(
        self,
        n_jobs: int,
        n_jobs_pairwise: int = 1,
        metric: str = "euclidean",
        p: int = 2,
        metric_params: Dict = None,
    ):
        """Contact map initialization

        Args:
            n_jobs (int, optional): number of cores to carry out the computation.
            n_jobs_pairwise (int, optional): allows to parallelize pairwise metric
                computations within a sample. Could be useful for large graphs, but
                will result in n_jobs*n_jobs_pairwise CPU core required. Defaults to 1.
            metric (str, optional): metric used to compute distance. Defaults to "euclidean".
            p (int, optional): power of minkowski distance. Defaults to 2.
            metric_params (Dict, optional): additional metrics parameters to pass. Defaults to None.
        """
        self.n_jobs = n_jobs
        self.n_jobs_pairwise = n_jobs_pairwise
        self.metric = metric

    def fit(self, proteins: List[Protein]) -> List[Protein]:
        """required for sklearn compatibility"""
        return proteins

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        """required for sklearn compatibility"""
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> List[Protein]:
        """Extract contact map from set of coordinates.

        Args:
            proteins: List[Protein]: List of arrays containing coordinates.

        Returns:
            List[Protein]): List of proteins with included contact maps
        """

        def compute_contact_map(protein: Protein) -> Protein:
            """Computes contact map between all the amino acids in X

            Args:
                X (pd.DataFrame): coordinates from a protein

            Returns:
                nx.Graph: labeled, fully connected weighted graph of the contact
                map.
            """
            protein.contact_map = pairwise_distances(
                protein.coordinates,
                metric=self.metric,
                n_jobs=self.n_jobs_pairwise,
            )
            return protein

        proteins = distribute_function(
            compute_contact_map,
            proteins,
            self.n_jobs,
            "Extracting contact map",
        )

        return proteins


class KNNGraph(GraphConstruction):
    """Extract KNN graph"""

    def __init__(
        self,
        n_neighbors: int,
        n_jobs: int,
        mode: str = "connectivity",
        metric: str = "euclidean",
        p: int = 2,
        metric_params: Dict = None,
    ):
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self.mode = mode
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, proteins: List[Protein]) -> List[Protein]:
        """required for sklearn compatibility"""
        return proteins

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        """required for sklearn compatibility"""
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> List[Protein]:
        """Extracts k-nearest neightbour graph from input contact map."""
        proteins = check_graphs(proteins)

        def knn_graph_func(protein: Protein) -> nx.Graph:
            """knn graph extraction function for each graph

            Args:
                proteins (Protein): graph to get the knn graph from

            Returns:
                Protein: proteins with knn adjancency matrix set.
            """
            protein.knn_adj = sparse.csr_matrix(
                kneighbors_graph(
                    protein.contact_map,
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    p=self.p,
                    metric_params=self.metric_params,
                    mode=self.mode,
                    include_self=False,
                )
            )
            protein.set_nx_graph(graph_type="knn_graph")
            return protein

        proteins = distribute_function(
            knn_graph_func,
            proteins,
            self.n_jobs,
            "Extracting KNN graph from contact map",
        )
        return proteins


class EpsilonGraph(GraphConstruction):
    """Extract epsilon graph"""

    def __init__(self, epsilon: float, n_jobs: int):
        self.epsilon = epsilon
        self.n_jobs = n_jobs

    def fit(self, proteins: List[Protein]) -> List[Protein]:
        """required for sklearn compatibility"""
        return proteins

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        """required for sklearn compatibility"""
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> List[Protein]:
        """Extracts epsilon graph from input contact map.

        Args:
            proteins List[Protein]: graph to get the epsilon graph from

        Returns:
            List[Protein]: list of epsilon graphs
        """
        proteins = check_graphs(proteins)

        def epsilon_graph_func(protein: Protein) -> nx.Graph:
            """epsilon graph extraction function for each graph

            Args:
                protein (Protein): input protein

            Returns:
                Protein: protein with epsilon graph adjacency matrix set.
            """
            protein.eps_adj = sparse.csr_matrix(
                np.where(protein.contact_map < self.epsilon, 1, 0)
            )
            protein.set_nx_graph(graph_type="eps_graph")
            return protein

        proteins = distribute_function(
            epsilon_graph_func,
            proteins,
            self.n_jobs,
            "Extracting Epsilon graph from contact map",
        )

        return proteins
