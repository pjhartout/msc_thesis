# -*- coding: utf-8 -*-

"""graphs.py

Handles graph extraction from a set of atom coordinates.

TODO: check docstrings
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

from proteinggnnmetrics.protein import Protein

from .constants import N_JOBS
from .utils.utils import tqdm_joblib
from .utils.validation import check_graphs


class GraphConstruction(metaclass=ABCMeta):
    """Defines skeleton of graph construction classes"""

    def __init__(self):
        pass

    def construct(self, samples: Union[List, np.ndarray]) -> np.ndarray:
        """Apply transformation

        Args:
            graphs (Any): list or array of samples

        Returns:
            np.ndarray: array of graphs
        """
        pass


class ContactMap(GraphConstruction):
    """Extract contact map from pdb file (fully connected weighted graph)

    Args:
        GraphConstruction (type): class constructor
    """

    def __init__(
        self,
        n_jobs: int = N_JOBS,
        n_jobs_pairwise: int = 1,
        metric: str = "euclidean",
        p: int = 2,
        metric_params: Dict = None,
        **kwargs,
    ):
        """Contact map initialization

        Args:
            n_jobs (int, optional): number of cores to carry out the computation. Defaults to N_JOBS.
            n_jobs_pairwise (int, optional): allows to parallelize pairwise metric
                computations within a sample. Could be useful for large graphs, but
                will result in n_jobs*n_jobs_pairwise CPU core required. Defaults to 1.
            metric (str, optional): metric used to compute distance. Defaults to "euclidean".
            p (int, optional): power of minkowski distance. Defaults to 2.
            metric_params (Dict, optional): additional metrics parameters to pass. Defaults to None.
        """
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
        self.n_jobs_pairwise = n_jobs_pairwise
        self.metric = metric

    def construct(self, proteins: List[Protein]) -> List[Protein]:
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

        with tqdm_joblib(
            tqdm(desc=f"Extracting contact map", total=len(proteins),)
        ) as progressbar:
            proteins = Parallel(n_jobs=N_JOBS)(
                delayed(compute_contact_map)(sample) for sample in proteins
            )

        return proteins


class KNNGraph(GraphConstruction):
    """Extract KNN graph
    """

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

    def construct(self, proteins: List[Protein]) -> List[Protein]:
        """Extracts k-nearest neightbour graph from input contact map.
        """
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
            return protein

        with tqdm_joblib(
            tqdm(
                desc=f"Extracting KNN graph from contact map",
                total=len(proteins),
            )
        ) as progressbar:
            proteins = Parallel(n_jobs=self.n_jobs)(
                delayed(knn_graph_func)(sample) for sample in proteins
            )
        return proteins


class EpsilonGraph(GraphConstruction):
    """Extract epsilon graph"""

    def __init__(
        self, epsilon: float = 3.0, n_jobs: int = N_JOBS, **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_jobs = n_jobs

    def construct(self, proteins: List[Protein]) -> List[Protein]:
        """Extracts epsilon graph from input contact map.

        Args:
            proteins List[Protein]: graph to get the epsilon graph from

        Returns:
            List[Protein]: list of epsilon graphs
        """
        proteins = check_graphs(proteins)

        def epsilon_graph_func_(protein: Protein) -> nx.Graph:
            """epsilon graph extraction function for each graph

            Args:
                protein (Protein): input protein

            Returns:
                Protein: protein with epsilon graph adjacency matrix set.
            """
            protein.eps_adj = sparse.csr_matrix(
                np.where(protein.contact_map < self.epsilon, 1, 0)
            )
            return protein

        with tqdm_joblib(
            tqdm(desc=f"Extracting epsilon graph", total=len(proteins),)
        ) as progressbar:
            proteins = Parallel(n_jobs=self.n_jobs)(
                delayed(epsilon_graph_func_)(sample) for sample in proteins
            )
        return proteins
