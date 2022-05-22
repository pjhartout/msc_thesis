#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""reps.py

This script computes all possible representations of a protein available.

Format of directory structure: /data/systematic/representations/organism/perturbation/twist/amount/*.pkl
"""

import logging
import os
import random
from enum import unique
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import hydra
import numpy as np
import pandas as pd
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here
from tqdm import tqdm

from proteinggnnmetrics.descriptors import (
    ESM,
    ClusteringHistogram,
    DegreeHistogram,
    DistanceHistogram,
    LaplacianSpectrum,
    RamachandranAngles,
    TopologicalDescriptor,
)
from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph, KNNGraph
from proteinggnnmetrics.kernels import LinearKernel
from proteinggnnmetrics.loaders import list_pdb_files, load_graphs
from proteinggnnmetrics.paths import ECOLI_PROTEOME, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates, Sequence
from proteinggnnmetrics.perturbations import (
    GaussianNoise,
    Mutation,
    Shear,
    Taper,
    Twist,
)
from proteinggnnmetrics.protein import Protein
from proteinggnnmetrics.utils.functions import (
    flatten_lists,
    remove_fragments,
    save_obj,
)

log = logging.getLogger(__name__)


def get_longest_protein_dummy_sequence(
    proteins: List[Protein], cfg: DictConfig
) -> int:
    seq = Sequence(n_jobs=cfg.compute.n_jobs)
    paths = [protein.path for protein in proteins]
    sequences = seq.fit_transform(paths)
    longest_sequence = max([len(protein.sequence) for protein in sequences])
    return longest_sequence


def load_unperturbed_proteins(cfg: DictConfig, perturbed: bool) -> List[Path]:
    """Loads unique unperturbed protein file names

    Args:
        cfg (DictConfig): Configuration.

    Returns:
        List[Path]: unique proteins seen across runs
    """
    protein_sets = []
    for file in os.listdir(here() / cfg.meta.splits_dir):
        if perturbed:
            if "_perturbed" in file:
                protein_sets.append(
                    pd.read_csv(here() / cfg.meta.splits_dir / file, sep="\t")
                )
        else:
            if "_unperturbed" in file:
                protein_sets.append(
                    pd.read_csv(here() / cfg.meta.splits_dir / file, sep="\t")
                )
    unique_protein_fnames = pd.concat(protein_sets).drop_duplicates()
    unique_protein_fnames = [
        HUMAN_PROTEOME / fname
        for fname in unique_protein_fnames.pdb_id.tolist()
    ]
    log.info(
        f"Found {len(unique_protein_fnames)} unique proteins in each set.."
    )
    return unique_protein_fnames  # type: ignore


def compute_basic_reps(
    cfg: DictConfig,
    protein_sets: List[Path],
    perturbation: Union[Tuple[str, Any], None] = None,
) -> List[Protein]:
    """Compute basic representations of proteins.

    Args:
        cfg (DictConfig): Configuration.
        protein_sets (pd.Series): List of proteins.

    Returns:
        list: List of proteins.

    """
    log.info("Computing basic representations.")
    basic_representations_steps = [
        (
            "coordinates backbone",
            Coordinates(
                granularity="backbone",
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "contact map",
            ContactMap(metric="euclidean", n_jobs=cfg.compute.n_jobs),
        ),
    ]
    if perturbation is not None:
        basic_representations_steps = [
            basic_representations_steps[0],
            perturbation,
            basic_representations_steps[1],
        ]
    proteins = pipeline.Pipeline(basic_representations_steps).fit_transform(
        protein_sets
    )
    log.info("Done computing basic representations.")
    return proteins  # type: ignore


def compute_esm_descriptors(
    cfg: DictConfig, proteins: List[Protein]
) -> List[Protein]:

    log.info("Computing ESM descriptors.")
    esm_steps = [
        (
            "compute_esm",
            ESM(
                size="M",
                longest_sequence=get_longest_protein_dummy_sequence(
                    proteins, cfg
                ),
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
    ]
    proteins = pipeline.Pipeline(esm_steps).fit_transform(proteins)
    log.info("Done computing ESM descriptors.")
    return proteins


def compute_tda_descriptors(
    cfg: DictConfig, proteins: List[Protein]
) -> List[Protein]:
    log.info("Computing TDA descriptors.")
    tda_steps = [
        (
            "tda",
            TopologicalDescriptor(
                "diagram",
                epsilon=0.01,
                n_bins=100,
                order=2,
                n_jobs=cfg.compute.n_jobs,
                landscape_layers=1,
                verbose=cfg.debug.verbose,
            ),
        ),
    ]
    proteins = pipeline.Pipeline(tda_steps).fit_transform(proteins)
    log.info("Done computing TDA descriptors.")
    return proteins


def compute_dihedral_angles(
    cfg: DictConfig, proteins: List[Protein]
) -> List[Protein]:
    """Compute dihedral angles.

    Args:
        cfg (DictConfig): Configuration.
        proteins (List[Protein]): List of proteins.

    Returns:
        list: List of proteins.

    """
    log.info("Computing dihedral angles.")
    dihedral_angles_steps = [
        (
            "dihedral angles",
            RamachandranAngles(
                from_pdb=False,
                n_bins=100,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
    ]
    proteins = pipeline.Pipeline(dihedral_angles_steps).fit_transform(proteins)
    log.info("Done computing dihedral angles.")
    return proteins


def compute_distance_histogram(
    cfg: DictConfig, proteins: List[Protein]
) -> List[Protein]:
    """Compute distance histogram.

    Args:
        cfg (DictConfig): Configuration.
        proteins (List[Protein]): List of proteins.

    Returns:
        list: List of proteins.

    """
    log.info("Computing distance histogram.")
    distance_histogram_steps = [
        (
            "distance histogram",
            DistanceHistogram(
                n_bins=100,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
    ]
    proteins = pipeline.Pipeline(distance_histogram_steps).fit_transform(
        proteins
    )
    log.info("Done computing distance histogram.")
    return proteins


def compute_eps_graphs(
    cfg: DictConfig,
    proteins: List[Protein],
    graph_descriptor_steps: List[Tuple[str, Any]],
    perturbation,
) -> None:
    log.info("Computing epsilon graphs.")
    for eps_value in cfg.meta.representations[0]["eps_graphs"]:
        log.info(f"Computing epsilon graphs for epsilon={eps_value}")
        eps_graph_steps = graph_descriptor_steps.copy()
        eps_graph_steps.insert(
            0,
            (
                "epsilon graph",
                EpsilonGraph(
                    epsilon=eps_value,
                    n_jobs=cfg.compute.n_jobs,
                    verbose=cfg.debug.verbose,
                ),
            ),
        )

        rep_specific = pipeline.Pipeline(eps_graph_steps).fit_transform(
            proteins
        )
        save_graphs(cfg, rep_specific, perturbation, "eps", eps_value)

    log.info("Done computing epsilon graphs.")


def compute_knn_graphs(
    cfg: DictConfig,
    proteins: List[Protein],
    graph_descriptor_steps: List[Tuple[str, Any]],
    perturbation,
):
    log.info("Computing KNN graphs.")
    for k in cfg.meta.representations[1]["knn_graphs"]:
        log.info(f"Computing KNN graphs for k={k}")
        knn_graph_steps = graph_descriptor_steps.copy()
        knn_graph_steps.insert(
            0,
            (
                "knn graph",
                KNNGraph(
                    n_neighbors=k,
                    n_jobs=cfg.compute.n_jobs,
                    verbose=cfg.debug.verbose,
                ),
            ),
        )

        rep_specific = pipeline.Pipeline(knn_graph_steps).fit_transform(
            proteins
        )
        save_graphs(cfg, rep_specific, perturbation, "knn", k)
    log.info("Done computing KNN graphs.")


def save_graphs(cfg, rep_specific, perturbation, graph_type, graph_param):
    save_obj(
        here()
        / cfg.paths.representations
        / cfg.paths.human
        / cfg.paths.perturbed
        / perturbation[0].split("_")[0]
        / graph_type
        / str(graph_param)
        / f"reps_{round(float(perturbation[0].split('_')[1]), 2)}.pkl",
        rep_specific,
    )


def compute_graphs(cfg: DictConfig, proteins: List[Protein], perturbation):
    """Compute graphs. Since this is parameter-dependent, we are going to create copies of protein datasets with the same basic attributes but different graph info that depends on the parameters.

    Args:
        cfg (DictConfig): Configuration.
        proteins (List[Protein]): List of proteins.

    Returns:
        list: List of proteins.

    """
    log.info("Computing graphs.")
    eps_graph_descriptor_steps = [
        (
            "degree_histogram",
            DegreeHistogram(
                graph_type="eps_graph",
                n_bins=100,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "clustering_coefficient",
            ClusteringHistogram(
                graph_type="eps_graph",
                n_bins=100,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "laplacian_spectrum",
            LaplacianSpectrum(
                graph_type="eps_graph", n_bins=100, n_jobs=cfg.compute.n_jobs,
            ),
        ),
    ]
    compute_eps_graphs(cfg, proteins, eps_graph_descriptor_steps, perturbation)

    knn_graph_descriptor_steps = [
        (
            "degree_histogram",
            DegreeHistogram(
                graph_type="knn_graph",
                n_bins=100,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "clustering_coefficient",
            ClusteringHistogram(
                graph_type="knn_graph",
                n_bins=100,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "laplacian_spectrum",
            LaplacianSpectrum(
                graph_type="knn_graph", n_bins=100, n_jobs=cfg.compute.n_jobs,
            ),
        ),
    ]
    compute_knn_graphs(cfg, proteins, knn_graph_descriptor_steps, perturbation)
    log.info("Done computing graphs.")


def compute_reps(
    cfg: DictConfig,
    protein_sets: List[Path],
    organism: str,
    perturbation: Union[None, Tuple[str, Any]] = None,
) -> None:
    proteins_basic_reps = compute_basic_reps(cfg, protein_sets, perturbation)

    proteins_constant = compute_dihedral_angles(cfg, proteins_basic_reps)
    proteins_constant = compute_distance_histogram(cfg, proteins_constant)
    # proteins_constant = compute_esm_descriptors(cfg, proteins_constant)
    # proteins_constant = compute_tda_descriptors(cfg, proteins_constant)
    log.info("Constant reps computed. Saving.")
    if organism == "human":
        if perturbation is None:
            save_obj(
                here()
                / cfg.paths.representations
                / cfg.paths.human
                / cfg.paths.unperturbed
                / "constant_reps.pkl",
                proteins_constant,
            )
        else:
            save_obj(
                here()
                / cfg.paths.representations
                / cfg.paths.human
                / cfg.paths.perturbed
                / perturbation[0].split("_")[0]
                / f"reps_{perturbation[0].split('_')[1]}.pkl",
                proteins_constant,
            )
    else:
        log.info("Patience.")

    log.info(
        "Computing representations depending on hyperparameters (i.e. graphs)."
    )
    compute_graphs(cfg, proteins_basic_reps, perturbation)
    log.info("Done computing unperturbed representations.")


def compute_reps_on_unperturbed_proteins(
    cfg: DictConfig, organism: str
) -> None:
    """Compute the unperturbed protein sets.

    Args:
        cfg (OmegaConf): Configuration.

    Returns:
        list: List of unperturbed protein sets.

    """
    log.info("Computing unperturbed protein sets.")
    protein_sets = load_unperturbed_proteins(cfg, perturbed=False)

    if cfg.debug.reduce_data:
        log.warning("Reducing data for debugging purposes.")
        protein_sets = protein_sets[: cfg.debug.sample_data_size]
    compute_reps(cfg, protein_sets, organism)


def gaussian_noise_perturbation(
    cfg: DictConfig, protein_sets: List[Path], organism: str
) -> None:
    """Compute the perturbed protein sets.

    Args:
        cfg (DictConfig): Configuration.
        protein_sets (List[Path]): List of protein sets.
    """
    log.info("Perturbing proteins with Gaussian noise.")
    for sigma in np.linspace(
        cfg.perturbations.gaussian_noise.min,
        cfg.perturbations.gaussian_noise.max,
        cfg.perturbations.n_perturbations,
    ):
        log.info(f"Gaussian level set to {sigma}.")
        perturbation = (
            f"gaussian_{sigma}",
            GaussianNoise(
                noise_mean=0,
                random_state=hash(
                    str(protein_sets)
                ),  # The seed is the same as long as the paths is the same.
                noise_variance=sigma,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        compute_reps(cfg, protein_sets, organism, perturbation)


def twist_perturbation(cfg, protein_sets, organism):
    """Compute the perturbed protein sets.

    Args:
        cfg (DictConfig): Configuration.
        protein_sets (List[Path]): List of protein sets.
    """
    log.info("Perturbing proteins with twist.")
    for alpha in np.linspace(
        cfg.perturbations.twist.min,
        cfg.perturbations.twist.max,
        cfg.perturbations.n_perturbations,
    ):
        log.info(f"Twist rad/Å set to {alpha}.")
        perturbation = (
            f"twist_{alpha}",
            Twist(
                alpha=alpha,
                random_state=hash(
                    str(protein_sets)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        compute_reps(cfg, protein_sets, organism, perturbation)


def shear_perturbation(cfg, protein_sets, organism):
    """Compute the perturbed protein sets.

    Args:
        cfg (DictConfig): Configuration.
        protein_sets (List[Path]): List of protein sets.
    """
    log.info("Perturbing proteins with shear.")
    for shear in np.linspace(
        cfg.perturbations.shear.min,
        cfg.perturbations.shear.max,
        cfg.perturbations.n_perturbations,
    ):
        log.info(f"Shear set to {shear} Å.")
        perturbation = (
            f"shear_{shear}",
            Shear(
                shear_x=shear,
                shear_y=shear,
                random_state=hash(
                    str(protein_sets)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        compute_reps(cfg, protein_sets, organism, perturbation)


def taper_perturbation(cfg, protein_sets, organism):
    """Compute the perturbed protein sets.

    Args:
        cfg (DictConfig): Configuration.
        protein_sets (List[Path]): List of protein sets.
    """
    log.info("Perturbing proteins with taper.")
    for taper in np.linspace(
        cfg.perturbations.taper.min,
        cfg.perturbations.taper.max,
        cfg.perturbations.n_perturbations,
    ):
        log.info(f"Taper set to {taper} Å.")
        perturbation = (
            f"taper_{taper}",
            Taper(
                a=taper,
                b=taper,
                random_state=hash(
                    str(protein_sets)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        compute_reps(cfg, protein_sets, organism, perturbation)


def compute_reps_on_perturbed_proteins(cfg, organism) -> None:
    log.info("Computing perturbed protein sets.")
    protein_sets = load_unperturbed_proteins(cfg, perturbed=True)

    if cfg.debug.reduce_data:
        protein_sets = protein_sets[: cfg.debug.sample_data_size]

    # Gaussian Noise perturbation
    gaussian_noise_perturbation(cfg, protein_sets, organism)
    # Twist perturbation
    twist_perturbation(cfg, protein_sets, organism)
    # Shear perturbation
    shear_perturbation(cfg, protein_sets, organism)
    # Taper perturbation
    taper_perturbation(cfg, protein_sets, organism)


@hydra.main(config_path=str(here()) + "/conf/", config_name="systematic")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Starting reps.py")
    for organism in ["human"]:
        # compute_reps_on_unperturbed_proteins(cfg, organism)
        compute_reps_on_perturbed_proteins(cfg, organism)
        log.info(f"Done with {organism}")
    log.info("Done computing representations")


if __name__ == "__main__":
    main()
