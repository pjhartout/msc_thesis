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
        pass

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
        noise_variance: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance

    def add_noise_to_protein(self, protein: Protein) -> Protein:
        noise = np.random.normal(
            loc=0, scale=self.noise_variance, size=len(protein.coordinates) * 3
        ).reshape(len(protein.coordinates), 3)
        protein.coordinates = protein.coordinates + noise
        return protein

    def fit(self, X: List[Protein]) -> None:
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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


class Mutation(GraphPerturbation):
    def __init__(self, p_mutate, **kwargs):
        super().__init__(**kwargs)
        self.p_mutate = p_mutate

    def mutate(self, protein: Protein) -> Protein:
        graph = protein.graphs[self.graph_type].copy()
        nodes_to_mutate = self.random_state.binomial(
            1, self.p_mutate, size=graph.number_of_nodes()
        )
        node_indices_to_mutate = np.where(nodes_to_mutate == 1.0)[0]
        for node_index in node_indices_to_mutate:
            node = graph.nodes[node_index]
            # Mutate node label
            node["residue"] = self.random_state.choice(AMINO_ACIDS)
        protein.graphs[self.graph_type] = graph
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


__all__ = [
    "GaussianNoise",
    "RemoveEdges",
    "AddEdges",
    "RewireEdges",
    "AddConnectedNodes",
    "Mutation",
]
