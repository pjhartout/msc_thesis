# -*- coding: utf-8 -*-

"""descriptors.py

Graph descriptors essentially extract fixed-length representation of a graph

"""

import re
from abc import ABCMeta, abstractmethod
from typing import Any, Iterable, List

import numpy as np

from .loaders import load_graphs
from .protein import Protein
from .utils.functions import distribute_function
from .utils.interval import Interval, cos, sin, square, stack

AMINO_ACIDS = [
    "GLU",
    "VAL",
    "LEU",
    "LYS",
    "ALA",
    "THR",
    "ASN",
    "GLY",
    "PHE",
    "ASP",
    "HIS",
    "MET",
    "TRP",
    "SER",
    "ILE",
    "ARG",
    "GLN",
    "CYS",
    "PRO",
    "TYR",
]


class Perturbation(metaclass=ABCMeta):
    """Defines skeleton of perturbation classes classes"""

    def __init__(self, random_state, n_jobs: int, verbose: bool):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, proteins: List[Protein]) -> None:
        """For pipeline compatibiltiy"""
        ...

    def transform(self, X: List[Protein]) -> List[Protein]:
        return X

    def fit_transform(self, X: List[Protein]) -> List[Protein]:
        """Apply perturbation to graph or graph representation"""
        return X


class GaussianNoise(Perturbation):
    """Adds Gaussian noise to coordinates"""

    def __init__(
        self,
        noise_mean: float,
        noise_std: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def add_noise_to_matrix(self, points):
        noise = np.random.normal(
            loc=0, scale=self.noise_std, size=len(points) * 3
        ).reshape(len(points), 3)
        points = points + noise
        return points

    def add_noise_to_protein(self, protein: Protein) -> Protein:
        if protein.coordinates.size > 0:
            protein.coordinates = self.add_noise_to_matrix(protein.coordinates)

        if protein.N_coordinates.size > 0:
            protein.N_coordinates = self.add_noise_to_matrix(
                protein.N_coordinates
            )
            protein.CA_coordinates = self.add_noise_to_matrix(
                protein.CA_coordinates
            )
            protein.C_coordinates = self.add_noise_to_matrix(
                protein.C_coordinates
            )

        return protein

    def fit(self, X: List[Protein]) -> None:
        ...

    def transform(self, X: List[Protein]) -> List[Protein]:
        return X

    def fit_transform(self, X: List[Protein], y=None) -> List[Protein]:
        X = distribute_function(
            self.add_noise_to_protein,
            X,
            self.n_jobs,
            "Adding Gaussian noise to proteins",
            show_tqdm=self.verbose,
        )
        return X


# The point cloud perturbation methods used here are inspired from
# https://github.com/eth-sri/3dcertify
class Twist(Perturbation):
    def __init__(
        self, alpha: float, random_state: int, n_jobs: int, verbose: bool
    ):
        super().__init__(random_state, n_jobs, verbose)
        self.alpha = alpha

    def twist_matrix(self, points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        x_transformed = x * cos(self.alpha * z) - y * sin(self.alpha * z)
        y_transformed = x * sin(self.alpha * z) + y * cos(self.alpha * z)
        z_transformed = z
        return stack(
            [x_transformed, y_transformed, z_transformed],
            axis=1,
            convert=isinstance(self.alpha, Interval),
        )

    def twist_protein(self, protein: Protein) -> Protein:
        if protein.coordinates.size > 0:
            protein.coordinates = self.twist_matrix(protein.coordinates)

        if protein.N_coordinates.size > 0:
            protein.N_coordinates = self.twist_matrix(protein.N_coordinates)
            protein.CA_coordinates = self.twist_matrix(protein.CA_coordinates)
            protein.C_coordinates = self.twist_matrix(protein.C_coordinates)

        return protein

    def fit(self, X: List[Protein]) -> None:
        ...

    def transform(self, X: List[Protein]) -> List[Protein]:
        return X

    def fit_transform(self, X: List[Protein], y=None) -> List[Protein]:
        X = distribute_function(
            self.twist_protein,
            X,
            self.n_jobs,
            "Twisting proteins",
            show_tqdm=self.verbose,
        )
        return X


class Shear(Perturbation):
    def __init__(
        self,
        shear_x: float,
        shear_y: float,
        random_state,
        n_jobs: int,
        verbose: bool,
    ):
        super().__init__(random_state, n_jobs, verbose)
        self.shear_x = shear_x
        self.shear_y = shear_y

    def shear_matrix(self, points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        x_transformed = self.shear_x * z + x
        y_transformed = self.shear_y * z + y
        z_transformed = z
        return stack(
            [x_transformed, y_transformed, z_transformed],
            axis=1,
            convert=isinstance(self.shear_x, Interval),
        )

    def shear_protein(self, protein: Protein) -> Protein:
        if protein.coordinates.size > 0:
            protein.coordinates = self.shear_matrix(protein.coordinates)

        if protein.N_coordinates.size > 0:
            protein.N_coordinates = self.shear_matrix(protein.N_coordinates)
            protein.CA_coordinates = self.shear_matrix(protein.CA_coordinates)
            protein.C_coordinates = self.shear_matrix(protein.C_coordinates)

        return protein

    def fit(self, X: List[Protein]) -> None:
        ...

    def transform(self, X: List[Protein]) -> List[Protein]:
        return X

    def fit_transform(self, X: List[Protein], y=None) -> List[Protein]:
        X = distribute_function(
            self.shear_protein,
            X,
            self.n_jobs,
            "Shearing proteins",
            show_tqdm=self.verbose,
        )
        return X


class Taper(Perturbation):
    def __init__(self, a, b, random_state, n_jobs: int, verbose: bool):
        super().__init__(random_state, n_jobs, verbose)
        self.a = a
        self.b = b

    def taper_matrix(self, points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        x_transformed = (0.5 * square(self.a) * z + self.b * z + 1) * x
        y_transformed = (0.5 * square(self.a) * z + self.b * z + 1) * y
        z_transformed = z
        return stack(
            [x_transformed, y_transformed, z_transformed],
            axis=1,
            convert=isinstance(self.a, Interval),
        )

    def taper_protein(self, protein: Protein) -> Protein:
        if protein.coordinates.size > 0:
            protein.coordinates = self.taper_matrix(protein.coordinates)

        if protein.N_coordinates.size > 0:
            protein.N_coordinates = self.taper_matrix(protein.N_coordinates)
            protein.CA_coordinates = self.taper_matrix(protein.CA_coordinates)
            protein.C_coordinates = self.taper_matrix(protein.C_coordinates)

        return protein

    def fit(self, X: List[Protein]) -> None:
        ...

    def transform(self, X: List[Protein]) -> List[Protein]:
        return X

    def fit_transform(self, X: List[Protein], y=None) -> List[Protein]:
        X = distribute_function(
            self.taper_protein,
            X,
            self.n_jobs,
            "Taper proteins",
            show_tqdm=self.verbose,
        )
        return X


class Mutation(Perturbation):
    def __init__(self, p_mutate, **kwargs):
        super().__init__(**kwargs)
        self.p_mutate = p_mutate

    def mutate(self, protein: Protein) -> Protein:
        seq = protein.sequence
        nodes_to_mutate = self.random_state.binomial(
            1, self.p_mutate, size=len(seq)
        )
        node_indices_to_mutate = np.where(nodes_to_mutate == 1.0)[0]
        for node_index in node_indices_to_mutate:
            # Mutate node label
            seq[node_index] = self.random_state.choice(AMINO_ACIDS)
        protein.sequence = seq
        return protein

    def fit(self, proteins: List[Protein]) -> None:
        return None

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> List[Protein]:
        """Apply perturbation to graph or graph representation"""
        proteins = distribute_function(
            self.mutate,
            proteins,
            self.n_jobs,
            "Mutation",
            show_tqdm=self.verbose,
        )
        return proteins


# Classes below are adapted from
# https://github.com/BorgwardtLab/ggme/blob/main/src/perturbations.py


class GraphPerturbation(Perturbation):
    """Base class for graph perturbations"""

    def __init__(self, graph_type, **kwargs) -> None:
        super().__init__(**kwargs)
        self.graph_type = graph_type


class RemoveEdges(GraphPerturbation):
    """Randomly remove edges."""

    def __init__(self, p_remove: float, **kwargs):
        """Remove edges with probability p_remove."""
        super().__init__(**kwargs)
        self.p_remove = p_remove

    def remove_edges(self, protein: Protein) -> Protein:
        """Apply perturbation."""
        graph = protein.graphs[self.graph_type].copy()
        edges_to_remove = self.random_state.binomial(
            1, self.p_remove, size=graph.number_of_edges()
        )
        edge_indices_to_remove = np.where(edges_to_remove == 1.0)[0]
        edges = list(graph.edges())

        for edge_index in edge_indices_to_remove:
            edge = edges[edge_index]
            graph.remove_edge(*edge)
        protein.graphs[self.graph_type] = graph
        return protein

    def fit(self, X: List[Protein]) -> None:
        ...

    def transform(self, X: List[Protein]) -> List[Protein]:
        return X

    def fit_transform(self, X: List[Protein], y=None) -> Any:
        X = distribute_function(
            self.remove_edges,
            X,
            self.n_jobs,
            "Removing edges",
            show_tqdm=self.verbose,
        )
        return X


class AddEdges(GraphPerturbation):
    """Randomly add edges."""

    def __init__(self, p_add: float, **kwargs):
        """Add edges with probability p_add."""
        super().__init__(**kwargs)
        self.p_add = p_add

    def add_edges(self, protein: Protein) -> Protein:
        """Apply perturbation."""
        graph = graph = protein.graphs[self.graph_type].copy()
        nodes = list(graph.nodes())

        for i, node1 in enumerate(nodes):
            nodes_to_connect = self.random_state.binomial(
                1, self.p_add, size=len(nodes)
            )
            nodes_to_connect[i] = 0  # Never introduce self connections
            node_idxs_to_connect = np.where(nodes_to_connect == 1)[0]
            for j in node_idxs_to_connect:
                node2 = nodes[j]
                graph.add_edge(node1, node2)

        protein.graphs[self.graph_type] = graph
        return protein

    def fit(self, X: List[Protein]) -> None:
        ...

    def transform(self, X: List[Protein]) -> List[Protein]:
        return X

    def fit_transform(self, X: List[Protein], y=None) -> List[Protein]:
        X = distribute_function(
            self.add_edges,
            X,
            self.n_jobs,
            "Adding edges",
            show_tqdm=self.verbose,
        )
        return X


class RewireEdges(GraphPerturbation):
    """Randomly rewire edges."""

    def __init__(self, p_rewire: float, **kwargs):
        """Rewire edges with probability p_rewire."""
        super().__init__(**kwargs)
        self.p_rewire = p_rewire

    def rewire_edges(self, protein: Protein) -> Protein:
        """Apply perturbation."""
        graph = protein.graphs[self.graph_type].copy()
        edges_to_rewire = self.random_state.binomial(
            1, self.p_rewire, size=graph.number_of_edges()
        )
        edge_indices_to_rewire = np.where(edges_to_rewire == 1.0)[0]
        edges = list(graph.edges())
        nodes = list(graph.nodes())

        for edge_index in edge_indices_to_rewire:
            edge = edges[edge_index]
            graph.remove_edge(*edge)

            # Randomly pick one of the nodes which should be detached
            if self.random_state.random() > 0.5:
                keep_node, detach_node = edge
            else:
                detach_node, keep_node = edge

            # Pick a random node besides detach node and keep node to attach to
            possible_nodes = list(
                filter(lambda n: n not in [keep_node, detach_node], nodes)
            )
            attach_node = self.random_state.choice(possible_nodes)
            graph.add_edge(keep_node, attach_node)

        protein.graphs[self.graph_type] = graph
        return protein

    def fit(self, X: List[Protein]) -> None:
        ...

    def transform(self, X: List[Protein]) -> List[Protein]:
        return X

    def fit_transform(self, X: List[Protein], y=None) -> Any:
        X = distribute_function(
            self.rewire_edges,
            X,
            self.n_jobs,
            "Rewiring edges",
            show_tqdm=self.verbose,
        )
        return X


class AddConnectedNodes(GraphPerturbation):
    """Randomly add nodes to graph."""

    def __init__(self, n_nodes: int, p_edge: float, **kwargs):
        """Add n_nodes nodes and attach edges to node with prob p_edge."""
        super().__init__(**kwargs)
        self.n_nodes = n_nodes
        self.p_edge = p_edge

    def add_connected_nodes(self, protein: Protein) -> Protein:
        """Apply perturbation."""
        graph = protein.graphs[self.graph_type].copy()

        for i in range(self.n_nodes):
            n_nodes = graph.number_of_nodes()
            # As the graphs are loaded from numpy arrays the nodes are simply
            # the index
            new_node = n_nodes
            graph.add_node(new_node)
            # Adding node label residue, sampling from available amino acids
            graph.nodes[new_node]["residue"] = self.random_state.choice(
                AMINO_ACIDS
            )
            nodes_idxs_to_attach = np.where(
                self.random_state.binomial(1, self.p_edge, size=n_nodes)
            )[0]
            for node in nodes_idxs_to_attach:
                graph.add_edge(new_node, node)

        protein.graphs[self.graph_type] = graph
        return protein

    def fit(self, X: List[Protein]) -> None:
        ...

    def transform(self, X: List[Protein]) -> List[Protein]:
        return X

    def fit_transform(self, X: List[Protein], y=None) -> Any:
        X = distribute_function(
            self.add_connected_nodes,
            X,
            self.n_jobs,
            "Adding connected nodes",
            show_tqdm=self.verbose,
        )
        return X


__all__ = [
    "GaussianNoise",
    "Twist",
    "Shear",
    "Taper",
    "RemoveEdges",
    "AddEdges",
    "RewireEdges",
    "AddConnectedNodes",
    "Mutation",
]
