# -*- coding: utf-8 -*-

"""protein.py

Protein function to store protein-related data.

The idea is to create an object holding all protein-specific data, keeping it
pre-computed if already computed.

TODO: docs, tests, citations, type hints., see if https://docs.python.org/3/library/dataclasses.html + Pydantic is useful
"""


import pickle
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import plotly.graph_objs as gobj
from matplotlib.pyplot import text
from scipy import sparse

from .utils.exception import AdjacencyMatrixError, GraphTypeError
from .utils.functions import flatten_lists


class Protein:
    """Main protein object to store data related to a protein

    TODO: make none of the elements in init required.
    """

    def __init__(
        self,
        name="",
        sequence="",
        coordinates=np.array([]),
        N_coordinates=np.array([]),
        C_coordinates=np.array([]),
        CA_coordinates=np.array([]),
        contact_map=np.array([]),
        knn_adj=np.array([]),
        eps_adj=np.array([]),
        knn_graph=None,
        eps_graph=None,
        contact_graph=None,
        degree_histogram=np.array([]),
        clustering_histogram=np.array([]),
        laplacian_spectrum_histogram=np.array([]),
        path=None,
        phi_psi_angles=np.array([]),
        verbose=False,
        interatomic_clashes=float,
    ):
        # Basic protein descriptors
        self.name = name
        self.path = path
        self.sequence = sequence

        # Processed data extracted from pdb
        self.coordinates = coordinates
        self.N_coordinates = N_coordinates
        self.C_coordinates = C_coordinates
        self.CA_coordinates = CA_coordinates
        self.contact_map = contact_map
        self.phi_psi_angles = phi_psi_angles
        self.interatomic_clashes = interatomic_clashes

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


@dataclass
class Graph(ABC):
    adjacency_matrix: npt.NDArray[Any]
    graph_type: str
    nx_graph: nx.Graph = field(init=False)
    sequence: str

    def __post_init__(self):
        self.nx_graph = self.build_nx_graph()

    @abstractmethod
    def build_nx_graph(self):
        if nx.is_empty(self.nx_graph):
            if sparse.issparse(self.adjacency_matrix):
                G = nx.from_scipy_sparse_matrix(self.adjacency_matrix)
            elif type(self.adjacency_matrix) == np.ndarray:
                G = nx.from_numpy_array(self.adjacency_matrix)
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
            return G


@dataclass
class WeisfeilerLehmanHash:
    """Weisfeiler-Lehman hash."""

    graph: nx.Graph
    n_iter: int
    node_label: str = "residue"
    hashes: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self.hashes = self.compute_hash()

    def compute_hash(self):
        """Computes Weisfeiler-Lehman hash of the protein."""
        hash_iter_0 = dict(
            Counter(list(dict(self.graph.nodes("residue")).values()))
        )
        hashes_iters = dict(
            Counter(
                flatten_lists(
                    list(
                        nx.weisfeiler_lehman_subgraph_hashes(
                            self.graph,
                            node_attr="residue",
                            iterations=self.n_iter,
                        ).values()
                    )
                )
            )
        )
        return hashes_iters | hash_iter_0


@dataclass
class Histogram(ABC):
    type: str
    n_bins: int


class ClusteringHistogram(Histogram):
    type: str = "clustering"
    n_bins: int


class DegreeHistogram(Histogram):
    type: str = "degree"
    n_bins: int


class LaplacianSpectrumHistogram(Histogram):
    type: str = "degree"
    n_bins: int


@dataclass
class ContactMap:
    map: npt.NDArray[Any]

    def plot(self, **kwargs):
        plt.imshow(self.map, **kwargs)
        plt.show()


@dataclass
class Protein:
    """Protein class.

    Attributes:
        graph: Graph object.
        contact_map: ContactMap object.
        weisfeiler_lehman_hash: WeisfeilerLehmanHash object.
        clustering_histogram: ClusteringHistogram object.
        degree_histogram: DegreeHistogram object.
        laplacian_spectrum_histogram: LaplacianSpectrumHistogram object.
    """

    graph: Graph
    contact_map: ContactMap = field(init=False)
    weisfeiler_lehman_hash: WeisfeilerLehmanHash = field(init=False)
    clustering_histogram: ClusteringHistogram = field(init=False)
    degree_histogram: DegreeHistogram = field(init=False)
    laplacian_spectrum_histogram: LaplacianSpectrumHistogram = field(
        init=False
    )

    def __post_init__(self):
        self.contact_map = self.build_contact_map()
        self.weisfeiler_lehman_hash = self.build_weisfeiler_lehman_hash()
        self.clustering_histogram = self.build_clustering_histogram()
        self.degree_histogram = self.build_degree_histogram()
        self.laplacian_spectrum_histogram = (
            self.build_laplacian_spectrum_histogram()
        )

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

    @abstractmethod
    def build_contact_map(self):
        """Builds contact map of the protein."""
        pass

    @abstractmethod
    def build_weisfeiler_lehman_hash(self):
        """Builds Weisfeiler-Lehman hash of the protein."""
        pass

    @abstractmethod
    def build_clustering_histogram(self):
        """Builds clustering histogram of the protein."""
        pass

    @abstractmethod
    def build_degree_histogram(self):
        """Builds degree histogram of the protein."""
        pass

    @abstractmethod
    def build_laplacian_spectrum_histogram(self):
        """Builds laplacian spectrum histogram of the protein."""
        pass


# class Protein:
#     """Main protein object to store data related to a protein"""

#     def __init__(
#         self,
#         name: Union[str, None],
#         sequence: str = "",
#         coordinates: Union[npt.NDArray[Any], None] = None,
#         contact_map: Union[npt.NDArray[Any], None] = None,
#         knn_adj: Union[npt.NDArray[Any], None] = None,
#         eps_adj: Union[npt.NDArray[Any], None] = None,
#         knn_graph: Union[nx.Graph, None] = None,
#         eps_graph: Union[nx.Graph, None] = None,
#         contact_graph: Union[nx.Graph, None] = None,
#         degree_histogram: Union[npt.NDArray[Any], None] = None,
#         clustering_histogram: Union[npt.NDArray[Any], None] = None,
#         laplacian_spectrum_histogram: Union[npt.NDArray[Any], None] = None,
#         verbose: bool = False,
#     ):
#         # Basic protein descriptors
#         self.name = name
#         self.sequence = sequence

#         # Processed data extracted from pdb
#         self.coordinates = coordinates
#         self.contact_map = contact_map

#         # Adjacency matrices
#         self.knn_adj = knn_adj
#         self.eps_adj = eps_adj

#         # Graph objects
#         self.graphs = {
#             "knn_graph": knn_graph,
#             "eps_graph": eps_graph,
#             "contact_graph": contact_graph,
#         }

#         # Histograms, depends on the graph used
#         self.descriptors = {
#             "knn_graph": {
#                 "degree_histogram": degree_histogram,
#                 "clustering_histogram": clustering_histogram,
#                 "laplacian_spectrum_histogram": laplacian_spectrum_histogram,
#                 "weisfeiler-lehman-hist": {},
#             },
#             "eps_graph": {
#                 "degree_histogram": degree_histogram,
#                 "clustering_histogram": clustering_histogram,
#                 "laplacian_spectrum_histogram": laplacian_spectrum_histogram,
#                 "weisfeiler-lehman-hist": {},
#             },
#             "contact_graph": {
#                 # TDA stuff
#             },
#         }
#         self.verbose = verbose

#     def set_nx_graph(self, graph_type: str):
#         """Returns graph of specified graph type with labeled nodes"""

#         def build_graph(adj_matrix):
#             # Return graph if already pre-computed
#             if self.graphs[graph_type] is not None:
#                 return self.graphs[graph_type]

#             else:
#                 if sparse.issparse(adj_matrix):
#                     G = nx.from_scipy_sparse_matrix(adj_matrix)
#                 elif type(adj_matrix) == np.ndarray:
#                     G = nx.from_numpy_array(adj_matrix)
#                 else:
#                     raise AdjacencyMatrixError(
#                         "Adjacency matrix in unexpected format. Please check the \
#                         adjacency matrix of the protein"
#                     )

#                 # Node mapping {0: "MET", etc...}
#                 node_labels_map = dict(
#                     zip(range(len(self.sequence)), self.sequence)
#                 )
#                 nx.set_node_attributes(G, node_labels_map, "residue")
#                 self.graphs[graph_type] = G

#         if graph_type == "knn_graph":
#             build_graph(self.knn_adj)

#         elif graph_type == "eps_graph":
#             build_graph(self.eps_adj)

#         elif graph_type == "contact_map":
#             build_graph(self.contact_map)

#         else:
#             raise GraphTypeError(
#                 'Wrong graph type specified, should be one of ["knn_graph", '
#                 '"epsilon_graph", "contact_map"]'
#             )

#     def set_weisfeiler_lehman_hashes(
#         self, graph_type: str, n_iter: int
#     ) -> None:
#         hash_iter_0 = dict(
#             Counter(
#                 list(dict(self.graphs[graph_type].nodes("residue")).values())
#             )
#         )
#         hashes = dict(
#             Counter(
#                 flatten_lists(
#                     list(
#                         nx.weisfeiler_lehman_subgraph_hashes(
#                             self.graphs[graph_type],
#                             node_attr="residue",
#                             iterations=n_iter,
#                         ).values()
#                     )
#                 )
#             )
#         )
#         self.descriptors[graph_type]["weisfeiler-lehman-hist"] = (
#             hashes | hash_iter_0
#         )

#     def save(self, path: PosixPath, auto_name: bool = True) -> None:

#         if auto_name:
#             path = path / str(self.name.split(".")[0] + ".pkl")

#         with open(path, "wb") as f:
#             pickle.dump(self, f)

#     def plot_contact_map(self):
#         plt.imshow(self.contact_map)
#         plt.colorbar()
#         plt.show()

#     def plot_point_cloud(self):
#         scene = {
#             "xaxis": {
#                 "title": "0th",
#                 "type": "linear",
#                 "showexponent": "all",
#                 "exponentformat": "e",
#             },
#             "yaxis": {
#                 "title": "1st",
#                 "type": "linear",
#                 "showexponent": "all",
#                 "exponentformat": "e",
#             },
#             "zaxis": {
#                 "title": "2nd",
#                 "type": "linear",
#                 "showexponent": "all",
#                 "exponentformat": "e",
#             },
#         }

#         fig = gobj.Figure()
#         fig.update_layout(scene=scene)

#         fig.add_trace(
#             gobj.Scatter3d(
#                 x=self.coordinates[:, 0],
#                 y=self.coordinates[:, 1],
#                 z=self.coordinates[:, 2],
#                 mode="markers",
#                 marker={
#                     "size": 4,
#                     "color": list(range(self.coordinates.shape[0])),
#                     "colorscale": "Viridis",
#                     "opacity": 0.8,
#                 },
#             )
#         )
#         return fig

#     @staticmethod
#     def plot_graph(
#         G: List[nx.Graph],
#         sample: int = 0,
#         with_labels: bool = True,
#         return_fig: bool = False,
#         node_size: int = 100,
#         fontsize: int = 5,
#     ):
#         pos = nx.spring_layout(G[sample])
#         fig = nx.draw_networkx(
#             G[sample],
#             pos=pos,
#             with_labels=False,
#             node_size=node_size,
#         )
#         for node, (x, y) in pos.items():
#             text(x, y, node, fontsize=fontsize, ha="center", va="center")

#         if return_fig:
#             return fig
#         else:
#             plt.show()

#     @staticmethod
#     def load(path: PosixPath):
#         with open(path, "rb") as f:
#             protein = pickle.load(f)
#         return protein
