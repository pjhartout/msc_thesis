#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""mean_distance_embedding.py

Mean distance embedding

"""


import argparse
import json
import logging
import os
import random
import sys
from enum import unique
from multiprocessing.sharedctypes import Value
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from proteinmetrics.descriptors import (
    ESM,
    ClusteringHistogram,
    DegreeHistogram,
    DistanceHistogram,
    LaplacianSpectrum,
    RamachandranAngles,
    TopologicalDescriptor,
)
from proteinmetrics.distance import MaximumMeanDiscrepancy
from proteinmetrics.graphs import ContactMap, EpsilonGraph, KNNGraph
from proteinmetrics.kernels import (
    LinearKernel,
    MultiScaleKernel,
    PersistenceFisherKernel,
)
from proteinmetrics.loaders import list_pdb_files, load_descriptor
from proteinmetrics.paths import DATA_HOME, ECOLI_PROTEOME, HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates, Sequence
from proteinmetrics.perturbations import (
    AddConnectedNodes,
    AddEdges,
    GaussianNoise,
    GraphPerturbation,
    Mutation,
    RemoveEdges,
    RewireEdges,
    Shear,
    Taper,
    Twist,
)
from proteinmetrics.protein import Protein
from proteinmetrics.utils.debug import SamplePoints
from proteinmetrics.utils.functions import (
    distribute_function,
    flatten_lists,
    make_dir,
    remove_fragments,
    save_obj,
)

log = logging.getLogger(__name__)


def load_proteins_from_config_run(
    cfg: DictConfig, perturbed: bool, run: int
) -> List[Path]:
    """Loads unique unperturbed protein file names"""
    protein_sets = list()
    if perturbed:
        protein_sets.append(
            pd.read_csv(
                here()
                / cfg.meta.splits_dir
                / f"data_split_{run}_perturbed.csv",
                sep="\t",
            )
        )
    else:
        protein_sets.append(
            pd.read_csv(
                here()
                / cfg.meta.splits_dir
                / f"data_split_{run}_unperturbed.csv",
                sep="\t",
            )
        )
    unique_protein_fnames = pd.concat(protein_sets).drop_duplicates()
    unique_protein_fnames = [
        HUMAN_PROTEOME / fname
        for fname in unique_protein_fnames.pdb_id.tolist()
    ]
    log.info(
        f"Found {len(unique_protein_fnames)} unique proteins in each set."
    )

    return unique_protein_fnames  # type: ignore


def handle_degree(proteins, high_eps=False):
    degree_histogram = load_descriptor(
        proteins, descriptor="degree_histogram", graph_type="eps_graph"
    )
    dists = pairwise_distances(degree_histogram, metric="euclidean")
    dists = pd.DataFrame(dists.flatten(), columns=["dists"])
    if not high_eps:
        dists.to_csv(
            here()
            / "exploring"
            / "mean_distance_embedding"
            / "mean_distance_embedding_degree_histogram.csv",
        )
    else:
        dists.to_csv(
            here()
            / "exploring"
            / "mean_distance_embedding"
            / "mean_distance_embedding_degree_histogram_high_eps.csv",
        )


def handle_clustering(proteins, high_eps=False):
    degree_histogram = load_descriptor(
        proteins, descriptor="clustering_histogram", graph_type="eps_graph"
    )
    dists = pairwise_distances(degree_histogram, metric="euclidean")
    dists = pd.DataFrame(dists.flatten(), columns=["dists"])
    if not high_eps:
        dists.to_csv(
            here()
            / "exploring"
            / "mean_distance_embedding"
            / "mean_distance_embedding_clustering_histogram.csv",
        )
    else:
        dists.to_csv(
            here()
            / "exploring"
            / "mean_distance_embedding"
            / "mean_distance_embedding_clustering_histogram_high_eps.csv",
        )


def handle_laplacian(proteins, high_eps=False):
    degree_histogram = load_descriptor(
        proteins,
        descriptor="laplacian_spectrum_histogram",
        graph_type="eps_graph",
    )
    dists = pairwise_distances(degree_histogram, metric="euclidean")
    dists = pd.DataFrame(dists.flatten(), columns=["dists"])
    if not high_eps:
        dists.to_csv(
            here()
            / "exploring"
            / "mean_distance_embedding"
            / "mean_distance_embedding_laplacian_spectrum.csv",
        )
    else:
        dists.to_csv(
            here()
            / "exploring"
            / "mean_distance_embedding"
            / "mean_distance_embedding_laplacian_spectrum_high_eps.csv",
        )


def handle_dihedral_histogram(proteins):
    dihedral_histogram = [protein.phi_psi_angles for protein in proteins]
    dists = pairwise_distances(dihedral_histogram, metric="euclidean")
    dists = pd.DataFrame(dists.flatten(), columns=["dists"])
    dists.to_csv(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "mean_distance_embedding_dihedral_hist.csv",
    )


def handle_dist_histogram(proteins):
    distance_hists = [protein.distance_hist for protein in proteins]
    dists = pairwise_distances(distance_hists, metric="euclidean")
    dists = pd.DataFrame(dists.flatten(), columns=["dists"])
    dists.to_csv(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "mean_distance_embedding_dist_hist.csv",
    )


def mean_distance_embedding(cfg: DictConfig,):
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(
                granularity="CA", n_jobs=cfg.compute.n_jobs, verbose=True
            ),
        ),
        (
            "contact map",
            ContactMap(
                metric="euclidean", n_jobs=cfg.compute.n_jobs, verbose=True
            ),
        ),
        (
            "egraph",
            EpsilonGraph(epsilon=8, n_jobs=cfg.compute.n_jobs, verbose=True),
        ),
        (
            "degree_histogram",
            DegreeHistogram(
                graph_type="eps_graph",
                n_bins=cfg.descriptors.degree_histogram.n_bins,
                bin_range=(
                    cfg.descriptors.degree_histogram.bin_range.min,
                    cfg.descriptors.degree_histogram.bin_range.max,
                ),
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "clustering",
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
                graph_type="eps_graph",
                n_bins=cfg.descriptors.laplacian_spectrum.n_bins,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "distance_histogram",
            DistanceHistogram(
                n_bins=cfg.descriptors.distance_histogram.n_bins,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "dihedral_histogram",
            RamachandranAngles(
                from_pdb=True,
                n_bins=cfg.descriptors.dihedral_anlges.n_bins,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
    ]

    # base_feature_pipe = pipeline.Pipeline(
    #     base_feature_steps, verbose=cfg.debug.verbose
    # )
    # proteins = load_proteins_from_config_run(cfg, perturbed=False, run=0)
    # proteins = base_feature_pipe.fit_transform(proteins)
    # handle_degree(proteins)
    # handle_clustering(proteins)
    # handle_laplacian(proteins)
    # handle_dist_histogram(proteins)
    # handle_dihedral_histogram(proteins)

    high_eps_feature_steps = [
        (
            "coordinates",
            Coordinates(
                granularity="CA", n_jobs=cfg.compute.n_jobs, verbose=True
            ),
        ),
        (
            "contact map",
            ContactMap(
                metric="euclidean", n_jobs=cfg.compute.n_jobs, verbose=True
            ),
        ),
        (
            "egraph",
            EpsilonGraph(epsilon=32, n_jobs=cfg.compute.n_jobs, verbose=True),
        ),
        (
            "degree_histogram",
            DegreeHistogram(
                graph_type="eps_graph",
                n_bins=cfg.descriptors.degree_histogram.n_bins,
                bin_range=(
                    cfg.descriptors.degree_histogram.bin_range.min,
                    cfg.descriptors.degree_histogram.bin_range.max,
                ),
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "clustering",
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
                graph_type="eps_graph",
                n_bins=cfg.descriptors.laplacian_spectrum.n_bins,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
    ]
    high_eps_feature_pipe = pipeline.Pipeline(
        high_eps_feature_steps, verbose=cfg.debug.verbose
    )
    proteins = load_proteins_from_config_run(cfg, perturbed=False, run=0)
    proteins_high_eps = high_eps_feature_pipe.fit_transform(proteins)
    handle_degree(proteins_high_eps, high_eps=True)
    handle_clustering(proteins_high_eps, high_eps=True)
    handle_laplacian(proteins_high_eps, high_eps=True)


@hydra.main(
    version_base=None,
    config_path=str(here()) + "/conf/",
    config_name="systematic",
)
def main(cfg: DictConfig):
    log.info("Starting mean_distance_embedding.py")
    log.info("Running with config:")
    log.info(OmegaConf.to_yaml(cfg))

    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    log.info("DATA_DIR")
    log.info(here())
    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Start with Weisfeiler-Lehman-based-experiments.
    # outside for loops for n_iters and k.

    mean_distance_embedding(cfg=cfg,)


if __name__ == "__main__":
    main()
