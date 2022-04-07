#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""large_scale_fastwlk.py

Large scale fastwlk test.

"""
from datetime import datetime
from tkinter import N
from typing import Dict

import hydra
import numpy as np
import plotly.graph_objects as go
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here

from proteinggnnmetrics.descriptors import DegreeHistogram, RamachandranAngles
from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.embeddings import ESM
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.kernels import KernelComposition, LinearKernel
from proteinggnnmetrics.loaders import (
    list_pdb_files,
    load_descriptor,
    load_graphs,
)
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.functions import networkx2grakel, positive_eig


@hydra.main(config_path=str(here()) + "/conf/", config_name="config.yaml")
def main(cfg: DictConfig):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
        ),
        (
            "contact map",
            ContactMap(metric="euclidean", n_jobs=cfg.compute.n_jobs),
        ),
        ("epsilon graph", EpsilonGraph(epsilon=4, n_jobs=cfg.compute.n_jobs)),
        (
            "rachmachandran angles",
            RamachandranAngles(
                from_pdb=True,
                n_bins=40,
                bin_range=(-np.pi, np.pi),
                verbose=cfg.verbose,
                n_jobs=cfg.compute.n_jobs,
            ),
        ),
        (
            "degree histogram",
            DegreeHistogram(
                n_bins=20,
                graph_type="eps_graph",
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.verbose,
            ),
        ),
    ]
    base_feature_pipeline = pipeline.Pipeline(base_feature_steps, verbose=True)
    proteins = base_feature_pipeline.fit_transform(pdb_files[:40])

    proteins_1 = proteins[:10]
    proteins_2 = proteins[10:]

    features_1 = [
        load_descriptor(proteins_1, "degree_histogram", "eps_graph"),
        np.array([protein.phi_psi_angles for protein in proteins_1]),
        load_graphs(proteins_1, "eps_graph"),
    ]

    features_2 = [
        load_descriptor(proteins_2, "degree_histogram", "eps_graph"),
        np.array([protein.phi_psi_angles for protein in proteins_2]),
        load_graphs(proteins_2, "eps_graph"),
    ]

    # K_XX = kernel_comp.compute_gram_matrix(features_1)
    # K_YY = kernel_comp.compute_gram_matrix(features_2)
    # K_XY = kernel_comp.compute_gram_matrix(features_1, features_2)

    mmd = MaximumMeanDiscrepancy(
        biased=True,
        kernel=KernelComposition(
            kernels=[
                WeisfeilerLehmanKernel(
                    n_jobs=cfg.compute.n_jobs,
                    precomputed=False,
                    n_iter=4,
                    node_label="residue",
                    biased=True,
                ),  # type: ignore
                LinearKernel(n_jobs=cfg.compute.n_jobs),
                LinearKernel(n_jobs=cfg.compute.n_jobs),
            ],
            composition_rule="product",
            kernel2reps=[2, 0, 1],
        ),
        verbose=True,
    ).compute(features_1, features_2)

    print(f"mmd: {mmd}")


if __name__ == "__main__":
    main()
