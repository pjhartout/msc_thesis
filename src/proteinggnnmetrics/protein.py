# -*- coding: utf-8 -*-

"""protein.py

Protein function to store protein-related data.

The idea is to create an object holding all protein-specific data, keeping it
pre-computed if already computed.

TODO: docs, tests, citations, type hints.
"""


import pickle
from typing import List

import networkx as nx
import numpy as np
from scipy import sparse

from .errors import AdjacencyMatrixError, GraphTypeError


class Protein:
    """Main protein object to store data related to a protein
    """

    def __init__(
        self,
        name=None,
        sequence: str = "",
        coordinates=None,
        contact_map=None,
        knn_adj=None,
        eps_adj=None,
        knn_graph=None,
        eps_graph=None,
        contact_graph=None,
        degree_histogram=None,
        clustering_histogram=None,
        laplacian_spectrum_histogram=None,
    ):
        # Basic protein descriptors
        self.name = name
        self.sequence = sequence

        # Processed data extracted from pdb
        self.coordinates = coordinates
        self.contact_map = contact_map

        # Adjacency matrices
        self.knn_adj = knn_adj
        self.eps_adj = eps_adj

        # Graph objects
        self.graphs = {
            "knn_graph": knn_graph,
            "eps_graph": eps_graph,
            "contact_graph": contact_graph,
        }

        # Histograms, depends on the graph used
        self.descriptors = {
            "knn_graph": {
                "degree_histogram": degree_histogram,
                "clustering_histogram": clustering_histogram,
                "laplacian_spectrum_histogram": laplacian_spectrum_histogram,
            },
            "eps_graph": {
                "degree_histogram": degree_histogram,
                "clustering_histogram": clustering_histogram,
                "laplacian_spectrum_histogram": laplacian_spectrum_histogram,
            },
            "contact_graph": {
                # TDA stuff
            },
        }

    def get_nx_graph(self, graph_type: str):
        """Returns graph of specified graph type with labeled nodes
        """

        def build_graph(adj_matrix):
            # Return graph if already pre-computed
            if self.graphs[graph_type] is not None:
                return self.graphs[graph_type]

            else:
                if sparse.issparse(adj_matrix):
                    G = nx.from_scipy_sparse_matrix(self.knn_adj)
                elif type(adj_matrix) == np.ndarray:
                    G = nx.from_numpy_array(self.knn_adj)
                else:
                    raise AdjacencyMatrixError(
                        "Adjacency matrix in unexpected format. Please check the \
                        adjacency matrix of the protein"
                    )

                # Node mapping {0: "MET", etc...}
                node_labels_map = dict(
                    zip(range(len(self.sequence)), self.sequence)
                )
                self.graphs[graph_type] = nx.relabel_nodes(G, node_labels_map)

        if graph_type == "knn_graph":
            build_graph(self.knn_adj)
        elif graph_type == "epsilon_graph":
            build_graph(self.eps_adj)
        elif graph_type == "contact_map":
            build_graph(self.contact_map)
        else:
            raise GraphTypeError(
                'Wrong graph type specified, should be one of ["knn_graph", '
                '"epsilon_graph", "contact_map"]'
            )
        return self.graphs[graph_type]

    def save(self, path, auto_name: bool = True):
        if auto_name:
            path = path + self.name.split(".")[0] + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def plot_point_cloud(self):
        pass

    def plot_contact_map(self):
        pass

    def plot_graph(self, graph_type):
        pass

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            protein = pickle.load(f)
        return protein