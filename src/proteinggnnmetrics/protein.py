# -*- coding: utf-8 -*-

"""protein.py

Protein function to store protein-related data.

The idea is to create an object holding all protein-specific data, keeping it
pre-computed if already computed.

TODO: docs, tests, citations, type hints., see if https://docs.python.org/3/library/dataclasses.html is useful
"""


import pickle
from collections import Counter
from pathlib import PosixPath
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objs as gobj
from matplotlib.pyplot import figure, text
from scipy import sparse

from .utils.exception import AdjacencyMatrixError, GraphTypeError
from .utils.functions import flatten_lists


class Protein:
    """Main protein object to store data related to a protein"""

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
        verbose: bool = False,
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
                "weisfeiler-lehman-hist": {},
            },
            "eps_graph": {
                "degree_histogram": degree_histogram,
                "clustering_histogram": clustering_histogram,
                "laplacian_spectrum_histogram": laplacian_spectrum_histogram,
                "weisfeiler-lehman-hist": {},
            },
            "contact_graph": {
                # TDA stuff
            },
        }
        self.verbose = verbose

    def set_nx_graph(self, graph_type: str):
        """Returns graph of specified graph type with labeled nodes"""

        def build_graph(adj_matrix):
            # Return graph if already pre-computed
            if self.graphs[graph_type] is not None:
                return self.graphs[graph_type]

            else:
                if sparse.issparse(adj_matrix):
                    G = nx.from_scipy_sparse_matrix(adj_matrix)
                elif type(adj_matrix) == np.ndarray:
                    G = nx.from_numpy_array(adj_matrix)
                else:
                    raise AdjacencyMatrixError(
                        "Adjacency matrix in unexpected format. Please check the \
                        adjacency matrix of the protein"
                    )

                # Node mapping {0: "MET", etc...}
                node_labels_map = dict(
                    zip(range(len(self.sequence)), self.sequence)
                )
                nx.set_node_attributes(G, node_labels_map, "residue")
                self.graphs[graph_type] = G

        if graph_type == "knn_graph":
            build_graph(self.knn_adj)

        elif graph_type == "eps_graph":
            build_graph(self.eps_adj)

        elif graph_type == "contact_map":
            build_graph(self.contact_map)

        else:
            raise GraphTypeError(
                'Wrong graph type specified, should be one of ["knn_graph", '
                '"epsilon_graph", "contact_map"]'
            )

    def set_weisfeiler_lehman_hashes(
        self, graph_type: str, n_iter: int
    ) -> None:
        hash_iter_0 = dict(
            Counter(
                list(dict(self.graphs[graph_type].nodes("residue")).values())
            )
        )
        hashes = dict(
            Counter(
                flatten_lists(
                    list(
                        nx.weisfeiler_lehman_subgraph_hashes(
                            self.graphs[graph_type],
                            node_attr="residue",
                            iterations=n_iter,
                        ).values()
                    )
                )
            )
        )
        self.descriptors[graph_type]["weisfeiler-lehman-hist"] = (
            hashes | hash_iter_0
        )

    def save(self, path: PosixPath, auto_name: bool = True) -> None:

        if auto_name:
            path = path / str(self.name.split(".")[0] + ".pkl")

        with open(path, "wb") as f:
            pickle.dump(self, f)

    def plot_contact_map(self):
        pass

    @staticmethod
    def plot_point_cloud(point_cloud):
        scene = {
            "xaxis": {
                "title": "0th",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e",
            },
            "yaxis": {
                "title": "1st",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e",
            },
            "zaxis": {
                "title": "2nd",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e",
            },
        }

        fig = gobj.Figure()
        fig.update_layout(scene=scene)

        fig.add_trace(
            gobj.Scatter3d(
                x=point_cloud[:, 0],
                y=point_cloud[:, 1],
                z=point_cloud[:, 2],
                mode="markers",
                marker={
                    "size": 4,
                    "color": list(range(point_cloud.shape[0])),
                    "colorscale": "Viridis",
                    "opacity": 0.8,
                },
            )
        )
        return fig

    @staticmethod
    def plot_graph(
        G: List[nx.Graph],
        sample: int = 0,
        with_labels: bool = True,
        return_fig: bool = False,
        node_size: int = 100,
        fontsize: int = 5,
    ):
        pos = nx.spring_layout(G[sample])
        fig = nx.draw_networkx(
            G[sample], pos=pos, with_labels=False, node_size=node_size,
        )
        for node, (x, y) in pos.items():
            text(x, y, node, fontsize=fontsize, ha="center", va="center")

        if return_fig:
            return fig
        else:
            plt.show()

    @staticmethod
    def load(path: PosixPath):
        with open(path, "rb") as f:
            protein = pickle.load(f)
        return protein
