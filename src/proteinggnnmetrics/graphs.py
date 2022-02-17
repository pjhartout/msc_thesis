# -*- coding: utf-8 -*-

"""graphs.py

Handles graph extraction from a set of atom coordinates.

"""

from abc import ABCMeta
from typing import Dict, List, Union

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

from .constants import N_JOBS
from .utils.utils import tqdm_joblib
from .utils.validation import check_graph


class GraphConstruction(metaclass=ABCMeta):
    """Defines skeleton of graph construction classes"""

    def __init__(self):
        pass

    def transform(self, samples: Union[List, np.ndarray]) -> np.ndarray:
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

    def transform(self, X: List) -> List:
        """Extract contact map from set of coordinates.

        Args:
            X (List): List of arrays containing coordinates.

        Returns:
            Xt (List): List of contact maps as square 2D matrices
        """

        def compute_contact_map(X: np.ndarray) -> np.ndarray:
            """Computes contact map between all the amino acids in X

            Args:
                X (np.ndarray): coordinates from a protein

            Returns:
                np.ndarray: square matrix of distance between all the residues in X.
            """
            return pairwise_distances(
                X, metric=self.metric, n_jobs=self.n_jobs_pairwise
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
    """Extract KNN graph

    Args:
        GraphConstruction (type): class constructor

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

    def transform(self, X: List) -> List:
        """Extracts k-nearest neightbour graph from input contact map.

        Args:
            X (List): list of contact maps

        Returns:
            List: list of k-NN graphs.
        """
        X = check_graph(X)

        def knn_graph_func(X: np.ndarray) -> np.ndarray:
            """knn graph extraction function for each graph

            Args:
                X (np.ndarray): graph to get the knn graph from

            Returns:
                np.ndarray:
            """
            return sparse.csr_matrix(
                kneighbors_graph(
                    X,
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    p=self.p,
                    metric_params=self.metric_params,
                    mode=self.mode,
                    include_self=False,
                )
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

    def transform(self, X: List) -> List:
        """Extracts epsilon graph from input contact map.

        Args:
            X (List): graph to get the epsilon graph from

        Returns:
            List: list of epsilon graphs
        """
        X = check_graph(X)

        def epsilon_graph_func_(X: np.ndarray) -> np.ndarray:
            """epsilon graph extraction function for each graph

            Args:
                X (np.ndarray): contact map for epsilon graph

            Returns:
                np.ndarray: epsilon graph
            """
            return sparse.csr_matrix(np.where(X < self.epsilon, 1, 0))

        with tqdm_joblib(
            tqdm(desc=f"Extracting epsilon graph", total=len(X),)
        ) as progressbar:
            Xt = Parallel(n_jobs=self.n_jobs)(
                delayed(epsilon_graph_func_)(sample) for sample in X
            )
        return Xt
