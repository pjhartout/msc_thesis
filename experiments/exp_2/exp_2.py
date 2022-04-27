#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""exp_2.py

The goal of this experiment is to investigate the effect of twisting on MMD features derived from TDA.

"""

import os
import pickle
import random

import hydra
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from gtda.homology import VietorisRipsPersistence
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinggnnmetrics.descriptors import (
    LaplacianSpectrum,
    TopologicalDescriptor,
)
from proteinggnnmetrics.distance import (
    MaximumMeanDiscrepancy,
    PearsonCorrelation,
    SpearmanCorrelation,
)
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.kernels import LinearKernel, PersistenceFisherKernel
from proteinggnnmetrics.loaders import (
    list_pdb_files,
    load_descriptor,
    load_graphs,
)
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.perturbations import GaussianNoise, Twist
from proteinggnnmetrics.utils.debug import SamplePoints, measure_memory, timeit
from proteinggnnmetrics.utils.functions import (
    flatten_lists,
    remove_fragments,
    tqdm_joblib,
)


def execute_run(cfg, run):
    os.makedirs(here() / cfg.experiments.results, exist_ok=True)
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    pdb_files = remove_fragments(pdb_files)
    sampled_files = random.Random(run).sample(
        pdb_files, cfg.experiments.sample_size * 2
    )
    midpoint = int(len(sampled_files) / 2)

    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
        ),
        (
            "contact_map",
            ContactMap(
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "epsilon_graph",
            EpsilonGraph(
                n_jobs=cfg.compute.n_jobs,
                epsilon=8,
                verbose=cfg.debug.verbose,
            ),
        ),
        # (
        #     "clustering_histogram",
        #     LaplacianSpectrum(
        #         graph_type="eps_graph",
        #         n_bins=100,
        #         n_jobs=cfg.compute.n_jobs,
        #         verbose=cfg.debug.verbose,
        #     ),
        # ),
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

    base_feature_pipeline = pipeline.Pipeline(
        base_feature_steps, verbose=cfg.debug.verbose
    )
    proteins = base_feature_pipeline.fit_transform(
        sampled_files[midpoint:],
    )

    results = list()
    for twist in tqdm(
        np.arange(
            cfg.experiments.perturbations.twist.min,
            cfg.experiments.perturbations.twist.max,
            cfg.experiments.perturbations.twist.step,
        ),
        position=1,
        leave=False,
        desc="Twist range",
    ):
        perturb_feature_steps = flatten_lists(
            [
                base_feature_steps[:1]
                + [
                    (
                        "twist",
                        Twist(
                            alpha=twist,
                            random_state=42,
                            n_jobs=cfg.compute.n_jobs,
                            verbose=cfg.debug.verbose,
                        ),
                    )
                ]
                + base_feature_steps[1:]
            ]
        )
        perturb_feature_pipeline = pipeline.Pipeline(
            perturb_feature_steps, verbose=cfg.debug.verbose
        )
        proteins_perturbed = perturb_feature_pipeline.fit_transform(
            sampled_files[:midpoint]
        )

        diagrams = [
            protein.descriptors["contact_graph"]["diagram"]
            for protein in proteins
        ]
        diagrams_perturbed = [
            protein.descriptors["contact_graph"]["diagram"]
            for protein in proteins_perturbed
        ]

        mmd_tda = MaximumMeanDiscrepancy(
            biased=True,
            squared=True,
            kernel=PersistenceFisherKernel(n_jobs=cfg.compute.n_jobs),  # type: ignore
        ).compute(diagrams, diagrams_perturbed)

        graphs = load_graphs(proteins, graph_type="eps_graph")
        graphs_perturbed = load_graphs(
            proteins_perturbed, graph_type="eps_graph"
        )

        mmd_wl = MaximumMeanDiscrepancy(
            biased=True,
            squared=True,
            kernel=WeisfeilerLehmanKernel(
                n_jobs=cfg.compute.n_jobs, biased=True
            ),  # type: ignore
        ).compute(graphs, graphs_perturbed)

        # spectrum = load_descriptor(
        #     proteins, "laplacian_spectrum_histogram", graph_type="eps_graph"
        # )
        # spectrum_perturbed = load_descriptor(
        #     proteins_perturbed,
        #     "laplacian_spectrum_histogram",
        #     graph_type="eps_graph",
        # )

        # mmd_wl = MaximumMeanDiscrepancy(
        #     biased=True,
        #     squared=True,
        #     kernel=LinearKernel(n_jobs=cfg.compute.n_jobs),  # type: ignore
        # ).compute(spectrum, spectrum_perturbed)

        results.append({"mmd_tda": mmd_tda, "mmd_wl": mmd_wl, "twist": twist})

    print("Dumping results")
    results = pd.DataFrame(results).to_csv(
        here() / cfg.experiments.results / "mmd_single_run_twist_{run}.csv"
    )


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf_2")
def main(cfg: DictConfig):
    os.makedirs(here() / cfg.experiments.results, exist_ok=True)
    with tqdm_joblib(
        tqdm(
            desc="Execute runs in parallel",
            total=len(list(range(cfg.compute.n_parallel_runs))),
        )
    ) as progressbar:
        Parallel(n_jobs=cfg.compute.n_parallel_runs)(
            delayed(execute_run)(cfg, run)
            for run in range(cfg.experiments.n_runs)
        )


if __name__ == "__main__":
    main()
