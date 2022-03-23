# -*- coding: utf-8 -*-

"""descriptors.py

Graph descriptors essentially extract fixed-length representation of a graph

TODO: check docstrings, citations
"""

from abc import ABCMeta
from tabnanny import verbose
from typing import Any, Callable, List, Tuple

import networkx as nx
import numpy as np
from gtda import curves, diagrams, homology, pipeline

from proteinggnnmetrics.loaders import load_descriptor

from .protein import Protein
from .utils.exception import TDAPipelineError
from .utils.functions import distribute_function


class Descriptor(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self, n_jobs):
        self.n_jobs = n_jobs

    def fit(self):
        """required for sklearn compatibility"""
        pass

    def transform(self):
        """required for sklearn compatibility"""
        pass

    def fit_transform(self):
        """Applies descriptor to graph"""
        pass


class DegreeHistogram(Descriptor):
    def __init__(self, graph_type: str, n_bins: int, n_jobs: int):
        self.n_bins = n_bins
        self.graph_type = graph_type
        self.n_jobs = n_jobs

    def fit(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> Any:
        def calculate_degree_histogram(protein: Protein, normalize=True):
            G = protein.graphs[self.graph_type]
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
            self.n_jobs,
            "Compute degree histogram",
        )
        return proteins


class ClusteringHistogram(Descriptor):
    def __init__(self, graph_type: str, n_jobs: int, normalize: bool = True):
        self.graph_type = graph_type
        self.n_jobs = n_jobs
        self.normalize = normalize

    def fit(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> Any:
        def calculate_degree_histogram(protein: Protein):
            G = protein.graphs[self.graph_type]
            degree_histogram = nx.degree_histogram(G)

            protein.descriptors[self.graph_type][
                "clustering_histogram"
            ] = degree_histogram

            return protein

        proteins = distribute_function(
            calculate_degree_histogram,
            proteins,
            self.n_jobs,
            "Compute degree histogram",
        )
        return proteins


class LaplacianSpectrum(Descriptor):
    def __init__(
        self,
        graph_type: str,
        n_bins: int,
        n_jobs: int,
        density: bool = False,
        bin_range: Tuple = (0, 2),
    ):
        self.graph_type = graph_type
        self.n_bins = n_bins
        self.density = density
        self.bin_range = bin_range
        self.n_jobs = n_jobs

    def fit(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> Any:
        def calculate_laplacian_spectrum(protein: Protein):
            G = protein.graphs[self.graph_type]
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
            self.n_jobs,
            "Compute Laplacian spectrum histogram",
        )
        return proteins


class TopologicalDescriptor(Descriptor):
    def __init__(
        self,
        tda_descriptor_type: str,
        epsilon: float,
        n_bins: int,
        order: int = 1,
        sigma: float = 0.01,
        weight_function: Callable = None,
        landscape_layers: int = None,
        n_jobs: int = None,
    ) -> None:
        self.tda_descriptor_type = tda_descriptor_type
        self.epsilon = epsilon
        self.sigma = sigma
        self.weight_function = weight_function
        self.n_bins = n_bins
        self.order = order
        self.landscape_layers = landscape_layers
        self.n_jobs = n_jobs
        self.base_tda_pipeline = [
            (
                "diagram",
                homology.VietorisRipsPersistence(
                    n_jobs=self.n_jobs, metric="precomputed"
                ),
            )
        ]
        self.tda_pipeline = [
            ("filter", diagrams.Filtering(epsilon=self.epsilon)),
            ("rescaler", diagrams.Scaler()),
        ]

    def fit(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> Any:

        if self.tda_descriptor_type == "diagram":
            self.tda_pipeline = self.base_tda_pipeline

        elif self.tda_descriptor_type == "landscape":
            self.tda_pipeline.extend(
                [
                    (
                        "landscape",
                        diagrams.PersistenceLandscape(
                            n_layers=self.landscape_layers,
                            n_bins=self.n_bins,
                            n_jobs=self.n_jobs,
                        ),
                    ),
                    (
                        "curves",
                        curves.StandardFeatures("max", n_jobs=self.n_jobs),
                    ),
                ]
            )

        elif self.tda_descriptor_type == "betti":
            self.tda_pipeline.extend(
                [
                    (
                        "betti",
                        diagrams.BettiCurve(
                            n_bins=self.n_bins, n_jobs=self.n_jobs
                        ),
                    ),
                    (
                        "derivative",
                        curves.Derivative(
                            order=self.order, n_jobs=self.n_jobs
                        ),
                    ),
                    (
                        "featurizer",
                        curves.StandardFeatures("max", n_jobs=self.n_jobs),
                    ),
                ]
            )

        elif self.tda_descriptor_type == "image":
            self.tda_pipeline.extend(
                [
                    (
                        "image",
                        diagrams.PersistenceImage(
                            sigma=0.1,
                            n_bins=self.n_bins,
                            weight_function=self.weight_function,
                            n_jobs=self.n_jobs,
                        ),
                    ),
                    (
                        "featureizer",
                        curves.StandardFeatures(
                            "identity", n_jobs=self.n_jobs
                        ),
                    ),
                ]
            )

        else:
            raise TDAPipelineError(
                'Wrong TDA pipeline specified, should be one of ["diagram", "landscape", "betti", "image"]'
            )

        contact_maps = [protein.contact_map for protein in proteins]
        diagram_data = pipeline.Pipeline(
            self.base_tda_pipeline, verbose=True
        ).fit_transform(contact_maps)
        if self.tda_descriptor_type != "diagram":
            tda_descriptors = pipeline.Pipeline(
                self.tda_pipeline, verbose=True
            ).fit_transform(diagram_data)

            for protein, diagram, tda_descriptor in zip(
                proteins, diagram_data, tda_descriptors
            ):
                protein.descriptors["contact_graph"][
                    self.tda_descriptor_type
                ] = tda_descriptor
                protein.descriptors["contact_graph"]["diagram"] = diagram

            return proteins
        else:
            for protein, diagram in zip(proteins, diagram_data):
                protein.descriptors["contact_graph"]["diagram"] = diagram

            return proteins
