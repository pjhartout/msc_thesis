# -*- coding: utf-8 -*-

"""protein.py

Protein function to store protein-related data.


TODO: docs, tests, citations, type hints.
"""


import pickle
from typing import List

import networkx as nx
import numpy as np
import scipy

from .errors import AdjacencyMatrixError, GraphTypeError


class Protein:
    """Main protein object to store data related to a protein
    """

    def __init__(
        self,
        name: str = "",
        sequence: List[str] = list(""),
        coordinates=None,
        contact_map=None,
        knn_adj=None,
        eps_adj=None,
    ):
        self.name = name
        self.sequence = sequence
        self.coordinates = coordinates
        self.contact_map = contact_map
        self.knn_adj = knn_adj
        self.eps_adj = eps_adj

    def get_nx_graph(self, graph_type: str = "knn_graph"):
        """Returns graph of specified graph type with labeled nodes
        """

        def build_graph(adj_matrix):
            if scipy.sparse.issparse(adj_matrix):
                G = nx.from_scipy_sparse_matrix(self.knn_adj)
            elif type(adj_matrix) == np.ndarray:
                G = nx.from_numpy_array(self.knn_adj)
            else:
                raise AdjacencyMatrixError(
                    "Adjacency matrix in unexpected format. Please check the adjacency matrix of the protein"
                )

            # Node mapping {0: "MET", etc...}
            node_labels_map = dict(
                zip(range(len(self.sequence)), self.sequence)
            )
            G = nx.relabel_nodes(G, node_labels_map)
            return G

        if graph_type == "knn_graph":
            G = build_graph(self.knn_adj)
        elif graph_type == "epsilon_graph":
            G = build_graph(self.eps_adj)
        elif graph_type == "contact_map":
            G = build_graph(self.contact_map)
        else:
            raise GraphTypeError(
                'Wrong graph type specified, should be one of ["knn_graph", "epsilon_graph", "contact_map"]'
            )
        return G

    @staticmethod
    def plot_point_cloud(protein):
        return protein

    @staticmethod
    def plot_contact_map(protein):
        return protein

    @staticmethod
    def save(path, protein):
        with open(path, "wb") as f:
            pickle.dump(f, protein)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            pickle.load(f)
