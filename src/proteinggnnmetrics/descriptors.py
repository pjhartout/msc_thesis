# -*- coding: utf-8 -*-

"""descriptors.py

Graph descriptors essentially extract fixed-length representation of a graph

TODO: check docstrings, citations
"""

from abc import ABCMeta
from typing import Any, List, Tuple

import networkx as nx
import numpy as np

from .constants import N_JOBS
from .protein import Protein
from .utils.utils import distribute_function


class Descriptor(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self, n_jobs):
        self.n_jobs = n_jobs

    def describe(self):
        """Applies descriptor to graph
        """
        pass


class DegreeHistogram(Descriptor):
    def __init__(self, graph_type: str, n_bins: int, n_jobs: int = N_JOBS):
        self.n_bins = n_bins
        self.graph_type = graph_type
        self.n_jobs = n_jobs

    def describe(self, proteins: List[Protein]) -> Any:
        def calculate_degree_histogram(protein: Protein, normalize=True):
            G = protein.get_nx_graph(self.graph_type)
            degrees = np.array([val for (node, val) in G.degree()])
            histogram = np.bincount(degrees, minlength=self.n_bins + 1)
            if normalize:
                histogram = histogram / np.sum(histogram)
            return histogram

        histograms = distribute_function(
            calculate_degree_histogram,
            proteins,
            "Compute degree histogram",
            self.n_jobs,
        )
        return histograms


class DegreeHistogram(Descriptor):
    def __init__(self, graph_type: str, n_bins: int, n_jobs: int = N_JOBS):
        self.n_bins = n_bins
        self.graph_type = graph_type
        self.n_jobs = n_jobs

    def describe(self, proteins: List[Protein]) -> Any:
        def calculate_degree_histogram(protein: Protein, normalize=True):
            G = protein.get_nx_graph(self.graph_type)
            degrees = np.array([val for (node, val) in G.degree()])
            histogram = np.bincount(degrees, minlength=self.n_bins + 1)

            if normalize:
                histogram = histogram / np.sum(histogram)

            protein.descriptors[self.graph_type][
                "degree_histogram"
            ] = histogram

            return protein

        proteins = distribute_function(
            calculate_degree_histogram,
            proteins,
            "Compute degree histogram",
            self.n_jobs,
        )
        return proteins


class ClusteringHistogram(Descriptor):
    def __init__(
        self, graph_type: str, n_bins: int, density: bool, n_jobs: int = N_JOBS
    ):
        self.graph_type = graph_type
        self.n_bins = n_bins
        self.density = density
        self.n_jobs = n_jobs

    def describe(self, proteins: List[Protein]) -> Any:
        def calculate_clustering_histogram(protein: Protein, normalize=True):
            G = protein.get_nx_graph(self.graph_type)
            coefficient_list = list(nx.clustering(G).values())
            histogram, _ = np.histogram(
                coefficient_list,
                bins=self.n_bins,
                range=(0.0, 1.0),
                density=self.density,
            )

            protein.descriptors[self.graph_type][
                "clustering_histogram"
            ] = histogram

            return protein

        proteins = distribute_function(
            calculate_clustering_histogram,
            proteins,
            "Compute degree histogram",
            self.n_jobs,
        )
        return proteins


class ClusteringHistogram(Descriptor):
    def __init__(
        self, graph_type: str, n_bins: int, density: bool, n_jobs: int = N_JOBS
    ):
        self.graph_type = graph_type
        self.n_bins = n_bins
        self.density = density
        self.n_jobs = n_jobs

    def describe(self, proteins: List[Protein]) -> Any:
        def calculate_clustering_histogram(protein: Protein, normalize=True):
            G = protein.get_nx_graph(self.graph_type)
            coefficient_list = list(nx.clustering(G).values())
            histogram, _ = np.histogram(
                coefficient_list,
                bins=self.n_bins,
                range=(0.0, 1.0),
                density=self.density,
            )

            protein.descriptors[self.graph_type][
                "clustering_histogram"
            ] = histogram

            return protein

        proteins = distribute_function(
            calculate_clustering_histogram,
            proteins,
            "Compute clustering histogram",
            self.n_jobs,
        )
        return proteins


class LaplacianSpectrum(Descriptor):
    def __init__(
        self,
        graph_type: str,
        n_bins: int,
        density: bool = False,
        bin_range: Tuple = (0, 2),
        n_jobs: int = N_JOBS,
    ):
        self.graph_type = graph_type
        self.n_bins = n_bins
        self.density = density
        self.bin_range = bin_range
        self.n_jobs = n_jobs

    def describe(self, proteins: List[Protein]) -> Any:
        def calculate_laplacian_spectrum(protein: Protein,):
            G = protein.get_nx_graph(self.graph_type)
            spectrum = nx.normalized_laplacian_spectrum(G)
            histogram = np.histogram(
                spectrum,
                bins=self.n_bins,
                density=self.density,
                range=self.bin_range,
            )

            protein.descriptors[self.graph_type][
                "laplacian_spectrum_histogram"
            ] = histogram

            return protein

        proteins = distribute_function(
            calculate_laplacian_spectrum,
            proteins,
            "Compute Laplacian spectrum histogram",
            self.n_jobs,
        )
        return proteins
