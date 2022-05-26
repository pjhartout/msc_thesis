# -*- coding: utf-8 -*-

"""descriptors.py

Graph descriptors essentially extract fixed-length representation of a graph

TODO: check docstrings, citations
"""


import os
import shutil
import uuid
import warnings
from abc import ABCMeta
from typing import Any, Callable, List, Tuple, Union

import esm
import networkx as nx
import numpy as np
import torch
from Bio.PDB import PDBParser, PPBuilder, vectors
from gtda import curves, diagrams, homology, pipeline
from tqdm import tqdm

from proteinmetrics.loaders import load_descriptor

from .paths import CACHE_DIR
from .protein import Protein
from .utils.exception import TDAPipelineError
from .utils.functions import (
    chunks,
    distribute_function,
    flatten_lists,
    load_obj,
    save_obj,
)

# Ignore warnings from networkx about a change they have not implemented yet.
warnings.simplefilter(action="ignore", category=FutureWarning)


class Descriptor(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self, n_jobs, verbose):
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, protein: List[Protein]) -> None:
        """required for sklearn compatibility"""
        ...

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        """required for sklearn compatibility"""
        return proteins

    def fit_transform(self, proteins: List[Protein]) -> List[Protein]:
        """Applies descriptor to graph"""
        return proteins


class DegreeHistogram(Descriptor):
    def __init__(
        self, graph_type: str, n_bins: int, n_jobs: int, verbose: bool = False
    ):
        self.n_bins = n_bins
        self.graph_type = graph_type
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, proteins: List[Protein]) -> None:
        ...

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> Any:
        def calculate_degree_histogram(protein: Protein, normalize=True):
            G = protein.graphs[self.graph_type]
            degrees = np.array([val for (node, val) in G.degree()])  # type: ignore
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
        n_bins: int,
        n_jobs: int,
        density: bool = True,
        verbose: bool = False,
    ):
        super().__init__(n_jobs, verbose)
        self.graph_type = graph_type
        self.n_bins = n_bins
        self.density = density

    def fit(self, proteins: List[Protein]) -> None:
        ...

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> Any:
        def calculate_degree_histogram(protein: Protein):
            G = protein.graphs[self.graph_type]
            coefficient_list = list(nx.clustering(G).values())
            hist, _ = np.histogram(
                coefficient_list,
                bins=self.n_bins,
                range=(0.0, 1.0),
                density=self.density,
            )

            protein.descriptors[self.graph_type]["clustering_histogram"] = hist

            return protein

        proteins = distribute_function(
            calculate_degree_histogram,
            proteins,
            self.n_jobs,
            "Compute clustering histogram",
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
        bin_range: Tuple[int, int] = (0, 2),
        verbose: bool = False,
    ):
        super().__init__(n_jobs, verbose)
        self.graph_type = graph_type
        self.n_bins = n_bins
        self.density = density
        self.bin_range = bin_range

    def fit(self, proteins: List[Protein]) -> None:
        ...

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> Any:
        def calculate_laplacian_spectrum(protein: Protein):
            G = protein.graphs[self.graph_type]
            spectrum = nx.normalized_laplacian_spectrum(G)
            hist, _ = np.histogram(
                spectrum,
                bins=self.n_bins,
                density=self.density,
                range=self.bin_range,
            )

            protein.descriptors[self.graph_type][
                "laplacian_spectrum_histogram"
            ] = hist

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
        homology_dimensions: Tuple = (0, 1, 2),
        order: int = 1,
        sigma: float = 0.01,
        weight_function: Union[None, Callable] = None,
        landscape_layers: Union[None, int] = None,
        n_jobs: int = 1,
        verbose: bool = False,
        use_caching: bool = True,
        n_chunks: int = 20,
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
        self.verbose = verbose
        self.use_caching = use_caching
        self.n_chunks = n_chunks

    def fit(self, proteins: List[Protein]) -> None:
        ...

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        return proteins

    def fit_transform(self, proteins: List[Protein], y=None) -> Any:

        if self.tda_descriptor_type == "diagram":
            ...

        elif self.tda_descriptor_type == "landscape":
            tda_pipeline = [
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

        elif self.tda_descriptor_type == "betti":
            tda_pipeline = [
                (
                    "betti",
                    diagrams.BettiCurve(
                        n_bins=self.n_bins, n_jobs=self.n_jobs
                    ),
                ),
                (
                    "derivative",
                    curves.Derivative(order=self.order, n_jobs=self.n_jobs),
                ),
                (
                    "featurizer",
                    curves.StandardFeatures("max", n_jobs=self.n_jobs),
                ),
            ]

        elif self.tda_descriptor_type == "image":
            tda_pipeline = [
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
                    curves.StandardFeatures("identity", n_jobs=self.n_jobs),
                ),
            ]

        else:
            raise TDAPipelineError(
                'Wrong TDA pipeline specified, should be one of ["diagram", "landscape", "betti", "image"]'
            )

        coordinates = [protein.coordinates for protein in proteins]
        if self.verbose:
            print("Starting Vietoris-Rips filtration process")

        def compute_persistence_diagram_for_point_cloud(coordinate):
            """Since I can't be bothered to fix giotto's implementation, I am parallelizing the process of computing diagrams for large collections of point clouds using my own function."""
            return homology.VietorisRipsPersistence(
                n_jobs=1,  # But we do this across n threads
                homology_dimensions=self.homology_dimensions,
            ).fit_transform(
                coordinate.reshape(1, coordinate.shape[0], coordinate.shape[1])
            )[
                0
            ]

        def load_diagram(path, diagram_cache):
            return load_obj(diagram_cache / path)

        if self.use_caching:
            if self.verbose:
                print("Caching to accelerate operation.")
            diagram_cache = (
                # The uuid is useful if this descriptor is called multiple
                # times
                CACHE_DIR
                / f"diagram_compute_cache_{uuid.uuid4().hex}/"
            )
            diagram_cache.mkdir(
                parents=True, exist_ok=True,
            )
            n_chunks = (
                int(len(coordinates) / self.n_chunks)
                + 1  # 20 is a good size of chunks
            )
            cks = chunks(coordinates, n_chunks)
            for i, ck in enumerate(cks):
                diagram_data = distribute_function(
                    compute_persistence_diagram_for_point_cloud,
                    ck,
                    n_jobs=self.n_jobs,
                    tqdm_label="Computing persistence diagrams in parallel",
                    show_tqdm=self.verbose,
                )
                save_obj(
                    diagram_cache / f"diagram_part_{i}.pkl", diagram_data,
                )
            diagram_data = list()

            diagram_data = distribute_function(
                load_diagram,
                os.listdir(diagram_cache),
                n_jobs=self.n_jobs,
                tqdm_label="Computing persistence diagrams in parallel",
                show_tqdm=self.verbose,
                diagram_cache=diagram_cache,
            )

            diagram_data = flatten_lists(diagram_data)
            shutil.rmtree(diagram_cache)
        else:
            diagram_data = distribute_function(
                compute_persistence_diagram_for_point_cloud,
                coordinates,
                n_jobs=self.n_jobs,
                tqdm_label="Computing persistence diagrams in parallel",
                show_tqdm=self.verbose,
            )

        if self.verbose:
            print(
                "Vietoris-Rips filtration process complete. Postprocessing..."
            )
        if self.tda_descriptor_type != "diagram":
            tda_descriptors = pipeline.Pipeline(
                tda_pipeline, verbose=self.verbose  # type: ignore
            ).fit_transform(diagram_data)

            for protein, diagram, tda_descriptor in zip(
                proteins, diagram_data, tda_descriptors
            ):
                protein.descriptors["contact_graph"][
                    self.tda_descriptor_type
                ] = tda_descriptor
                protein.descriptors["contact_graph"]["diagram"] = diagram
            if self.verbose:
                print("TDA descriptor computation complete")
            return proteins
        else:
            for protein, diagram in zip(proteins, diagram_data):
                protein.descriptors["contact_graph"]["diagram"] = diagram
            if self.verbose:
                print("TDA descriptor computation complete")
            return proteins


class RamachandranAngles(Descriptor):
    def __init__(
        self,
        from_pdb: bool,
        n_bins: int,
        n_jobs: int,
        verbose: bool,
        bin_range: Tuple[float, float] = (-np.pi, np.pi),
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
        structure = parser.get_structure(protein.path.stem, protein.path)  # type: ignore

        angles = dict()
        for idx_model, model in enumerate(structure):
            polypeptides = PPBuilder().build_peptides(model)
            for idx_poly, poly in enumerate(polypeptides):
                angles[f"{idx_model}_{idx_poly}"] = poly.get_phi_psi_list()

        phi = np.array(flatten_lists(angles.values()), dtype=object)[  # type: ignore
            :, 0
        ].astype(
            float
        )
        phi = np.histogram(
            phi[phi != None],
            bins=self.n_bins,
            density=self.density,
            range=self.bin_range,
        )[0]
        psi = np.array(flatten_lists(angles.values()), dtype=object)[  # type: ignore
            :, 1
        ].astype(
            float
        )
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
        phi, _ = np.histogram(
            np.array(phi_psi_angles, dtype=object)[:, 0][  # type: ignore
                np.array(phi_psi_angles, dtype=object)[:, 0] != None  # type: ignore
            ].astype(float),
            bins=self.n_bins,
            density=self.density,
            range=self.bin_range,
        )
        psi, _ = np.histogram(
            np.array(phi_psi_angles, dtype=object)[:, 1][  # type: ignore
                np.array(phi_psi_angles, dtype=object)[:, 1] != None  # type: ignore
            ].astype(float),
            bins=self.n_bins,
            density=self.density,
            range=self.bin_range,
        )
        protein.phi_psi_angles = np.concatenate((phi, psi))
        return protein

    def fit(self):
        ...

    def transform(self):
        ...

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


class DistanceHistogram(Descriptor):
    def __init__(
        self,
        n_bins,
        n_jobs: int,
        verbose,
        bin_range: Tuple[int, int] = (0, 300),
    ):
        super().__init__(n_jobs, verbose)
        self.n_bins = n_bins
        self.bin_range = bin_range

    def get_histogram(self, protein: Protein) -> Protein:
        """Gets the interatomic clashes"""
        triu = np.triu(protein.contact_map)
        triu = triu[triu != 0]  # Why not reduce the amount of processing by 2?
        hist, _ = np.histogram(
            triu, bins=self.n_bins, range=self.bin_range, density=True,
        )
        protein.distance_hist = hist
        return protein

    def fit(self):
        ...

    def transform(self):
        ...

    def fit_transform(self, proteins: List[Protein], y=None) -> List[Protein]:
        """Gets the distance histograms from the contact graph"""

        proteins = distribute_function(
            self.get_histogram,
            proteins,
            self.n_jobs,
            "Extracting interatomic distance_histogram",
            show_tqdm=self.verbose,
        )

        return proteins


class Embedding(metaclass=ABCMeta):
    def __init__(self, n_jobs, verbose) -> None:
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, sequences: List[Protein]) -> None:
        """Fit the embedding to the given sequences.

        Args:
            sequences (List[Protein]): list of sequences to embed.
        """
        ...

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        """Transform the given sequences to embeddings.

        Args:
            sequences (List[Protein]): list of sequences to embed.

        Returns:
            List[Protein]: list of embeddings for each sequence.
        """
        return proteins

    def fit_transform(self, proteins: List[Protein]) -> List[Protein]:
        """Fit the embedding to the given sequences and transform them.

        Args:
            sequences (List[Protein]): list of sequences to embed.

        Returns:
            List[Protein]: list of embeddings for each sequence.
        """
        return proteins


class ESM(Embedding):
    _size_options = ["M", "XL"]

    def __init__(
        self,
        size: str,
        longest_sequence: int,
        n_jobs: int,
        verbose: bool,
        n_chunks: int = 20,
    ) -> None:
        """Used for dummy to ensure all embeddings have the same size even when run on different sets of data

        Args:
            size (str): size of the model used for embeddings
            longest_sequence (int): dummy sequence used to make sure all       embeddings have the same size.
            n_jobs (int): number of threads to use
            verbose (bool): verbosity
        """
        super().__init__(n_jobs, verbose)
        self.size = size
        self.longest_sequence = longest_sequence
        self.n_chunks = n_chunks

    def fit(self, sequences: List[Protein], y=None) -> None:
        """Fit the embedding to the given sequences.

        Args:
            sequences (List[Protein]): list of sequences to embed.
        """
        ...

    def transform(self, sequences: List[Protein], y=None) -> List[Protein]:
        """Transform the given sequences to embeddings.

        Args:
            a (List[Protein]): list of sequences to embed.

        Returns:
            List[Protein]: list of embeddings for each sequence.
        """
        return sequences

    def fit_transform(self, proteins: List[Protein], y=None) -> List[Protein]:
        """Actual function used to compute embeddings of a list of
        proteins.
        The model is not fitted in this function but this is to ensure
        consistency with the rest of the library

        Args:
            sequences (List[Protein]): _description_

        Returns:
            List[Protein]: _description_
        """

        if self.verbose:
            print("Computing embeddings with ESM...")
            print("Loading model...")
        if self.size == "M":
            model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
            repr_layer = 6
        elif self.size == "XL":
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            repr_layer = 33
        else:
            raise RuntimeError(f"Size must be one of {self._size_options}",)
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        # TODO: See if distribution makes sense here.
        if self.verbose:
            print("Getting sequences...")
        sequences = [
            (protein.name, protein.sequence_as_str())
            for protein in tqdm(proteins, disable=not self.verbose)
        ]
        sequences.append(("dummy", "A" * self.longest_sequence))
        _, _, batch_tokens = batch_converter(sequences)
        if self.verbose:
            print("Computing embeddings...")

        cks = chunks(batch_tokens, self.n_chunks)

        reps = list()

        def execute_chunk(ck):
            with torch.no_grad():
                results = model(
                    ck, repr_layers=[repr_layer], return_contacts=False,
                )
            token_representations = results["representations"][repr_layer]
            return token_representations

        if self.verbose:
            total = divmod(len(proteins), self.n_chunks)[0]
            if total == 0:
                total = 1
            for ck in tqdm(cks, total=total):
                token_representations = execute_chunk(ck)
                reps.append(token_representations)
        else:
            for ck in cks:
                token_representations = execute_chunk(ck)
                reps.append(token_representations)

        token_representations = flatten_lists(reps)
        # Remove dummy embedding
        token_representations[-1]
        if self.verbose:
            print("Post-processing embeddings...")

        for protein, token_rep in zip(proteins, token_representations):
            protein.embeddings["esm"] = (
                token_rep[1 : len(protein.sequence) + 1].mean(0).numpy()
            )

        return proteins