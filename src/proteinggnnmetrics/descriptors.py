# -*- coding: utf-8 -*-

"""descriptors.py

Graph descriptors essentially extract fixed-length representation of a graph

TODO: check docstrings, citations
"""

from abc import ABCMeta
from ctypes import Union
from tabnanny import verbose
from typing import Any, Callable, List, Tuple

import networkx as nx
import numpy as np
from Bio.PDB import PDBParser, PPBuilder, vectors
from gtda import curves, diagrams, homology, pipeline

from proteinggnnmetrics.loaders import load_descriptor

from .protein import Protein
from .utils.exception import TDAPipelineError
from .utils.functions import distribute_function, flatten_lists


class Descriptor(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self, n_jobs, verbose):
        self.n_jobs = n_jobs
        self.verbose = verbose

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
    def __init__(
        self, graph_type: str, n_bins: int, n_jobs: int, verbose: bool = False
    ):
        self.n_bins = n_bins
        self.graph_type = graph_type
        self.n_jobs = n_jobs
        self.verbose = verbose

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
            show_tqdm=self.verbose,
        )
        return proteins


class ClusteringHistogram(Descriptor):
    def __init__(
        self,
        graph_type: str,
        n_jobs: int,
        normalize: bool = True,
        verbose: bool = False,
    ):
        super().__init__(n_jobs, verbose)
        self.graph_type = graph_type
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
            show_tqdm=self.verbose,
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
        verbose: bool = False,
    ):
        super().__init__(n_jobs, verbose)
        self.graph_type = graph_type
        self.n_bins = n_bins
        self.density = density
        self.bin_range = bin_range

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
            show_tqdm=self.verbose,
        )
        return proteins


class TopologicalDescriptor(Descriptor):
    def __init__(
        self,
        tda_descriptor_type: str,
        epsilon: float,
        n_bins: int,
        homology_dimensions: Tuple[int] = (1, 2, 3),
        order: int = 1,
        sigma: float = 0.01,
        weight_function: Callable = None,
        landscape_layers: int = None,
        n_jobs: int = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(n_jobs, verbose)
        self.tda_descriptor_type = tda_descriptor_type
        self.epsilon = epsilon
        self.sigma = sigma
        self.weight_function = weight_function
        self.n_bins = n_bins
        self.homology_dimensions = homology_dimensions
        self.order = order
        self.landscape_layers = landscape_layers
        self.n_jobs = n_jobs
        self.tda_pipeline = [
            (
                "diagram",
                homology.VietorisRipsPersistence(
                    n_jobs=self.n_jobs,
                    metric="precomputed",
                    homology_dimensions=self.homology_dimensions,
                ),
            ),
            ("filter", diagrams.Filtering(epsilon=self.epsilon)),
            ("rescaler", diagrams.Scaler()),
        ]
        self.verbose = verbose

    def fit(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> Any:

        if self.tda_descriptor_type == "diagram":
            pass

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
        tda_features = pipeline.Pipeline(
            self.tda_pipeline, verbose=True
        ).fit_transform(contact_maps)
        for protein, diagram, tda_descriptor in zip(
            proteins, diagram_data, tda_descriptors
        ):
            protein.descriptors["contact_graph"][
                self.tda_descriptor_type
            ] = tda_descriptor
            protein.descriptors["contact_graph"]["diagram"] = diagram

        return proteins


class RamachandranAngles(Descriptor):
    def __init__(
        self,
        from_pdb: bool,
        n_bins: int,
        bin_range: Tuple[float, float],
        n_jobs: int,
        verbose: bool,
        density: bool = True,
    ) -> None:
        super().__init__(n_jobs, verbose)
        self.from_pdb = from_pdb
        self.n_bins = n_bins
        self.density = density
        self.bin_range = bin_range

    def get_angles_from_pdb(self, protein: Protein) -> Protein:
        """Assumes only one chain"""
        parser = PDBParser()
        structure = parser.get_structure(protein.path.stem, protein.path)

        angles = dict()
        for idx_model, model in enumerate(structure):
            polypeptides = PPBuilder().build_peptides(model)
            for idx_poly, poly in enumerate(polypeptides):
                angles[f"{idx_model}_{idx_poly}"] = poly.get_phi_psi_list()

        phi = np.array(flatten_lists(angles.values()), dtype=object)[
            :, 0
        ].astype(float)
        phi = np.histogram(
            phi[phi != None],
            bins=self.n_bins,
            density=self.density,
            range=self.bin_range,
        )[0]
        psi = np.array(flatten_lists(angles.values()), dtype=object)[
            :, 1
        ].astype(float)
        psi = np.histogram(
            psi[psi != None],
            bins=self.n_bins,
            density=self.density,
            range=self.bin_range,
        )[0]
        protein.phi_psi_angles = np.concatenate([phi, psi])
        return protein

    def get_angles_from_coordinates(self, protein: Protein) -> Protein:
        """Gets the angles from the N, CA and C coordinates"""
        phi_psi_angles = list()
        for position in range(len(protein.sequence)):
            # Phi angle
            if position > 0:
                phi = (
                    vectors.calc_dihedral(
                        vectors.Vector(protein.C_coordinates[position - 1]),
                        vectors.Vector(protein.N_coordinates[position]),
                        vectors.Vector(protein.CA_coordinates[position]),
                        vectors.Vector(protein.C_coordinates[position]),
                    ),
                )[0]
            else:
                phi = None
            # Psi angle
            if position < len(protein.sequence) - 1:
                psi = (
                    vectors.calc_dihedral(
                        vectors.Vector(protein.N_coordinates[position]),
                        vectors.Vector(protein.CA_coordinates[position]),
                        vectors.Vector(protein.C_coordinates[position]),
                        vectors.Vector(protein.N_coordinates[position + 1]),
                    ),
                )[0]

            else:
                psi = None
            phi_psi_angles.append((phi, psi))
        phi = np.histogram(
            np.array(phi, dtype=object)[:, 0][
                np.array(phi)[:, 0] != None
            ].astype(float),
            bins=self.n_bins,
            density=self.density,
            range=self.bin_range,
        )
        psi = np.histogram(
            np.array(psi, dtype=object)[:, 1][
                np.array(psi)[:, 1] != None
            ].astype(float),
            bins=self.n_bins,
            density=self.density,
            range=self.bin_range,
        )
        protein.phi_psi_angles = np.concatenate((phi, psi))
        return protein

    def fit(self):
        pass

    def transform(self):
        pass

    def fit_transform(self, proteins: List[Protein], y=None) -> List[Protein]:
        """Gets the angles from the list of pdb files"""

        if self.from_pdb:
            proteins = distribute_function(
                self.get_angles_from_pdb,
                proteins,
                self.n_jobs,
                "Extracting Rachmachandran angles from pdb files",
                show_tqdm=self.verbose,
            )
        else:
            proteins = distribute_function(
                self.get_angles_from_coordinates,
                proteins,
                self.n_jobs,
                "Extracting Rachmachandran angles from coordinates",
                show_tqdm=self.verbose,
            )

        return proteins


class InteratomicClash(Descriptor):
    def __init__(self, threshold, n_jobs, verbose):
        super().__init__(n_jobs, verbose)
        self.threshold = threshold

    def get_clashes(self, protein: Protein) -> Protein:
        """Gets the interatomic clashes"""
        clashes = np.where(protein.contact_map < self.threshold, 1, 0)
        np.fill_diagonal(clashes, 0)
        protein.interatomic_clashes = clashes.sum() / (
            (len(protein.sequence)) ** 2 - len(protein.sequence)
        )
        return protein

    def fit(self):
        pass

    def transform(self):
        ...

    def fit_transform(self, proteins: List[Protein], y=None) -> List[Protein]:
        """Gets the angles from the list of pdb files"""

        proteins = distribute_function(
            self.get_clashes,
            proteins,
            self.n_jobs,
            "Extracting interatomic clashes",
            show_tqdm=self.verbose,
        )

        return proteins
