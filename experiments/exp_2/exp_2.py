#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""exp_2.py

The goal of this experiment is to investigate the effect of twisting on MMD features derived from TDA.

"""

import os
import pickle

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

from proteinggnnmetrics.descriptors import TopologicalDescriptor
from proteinggnnmetrics.distance import (
    MaximumMeanDiscrepancy,
    PearsonCorrelation,
    SpearmanCorrelation,
)
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.kernels import PersistenceFisherKernel
from proteinggnnmetrics.loaders import list_pdb_files, load_graphs
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.perturbations import GaussianNoise, Twist
from proteinggnnmetrics.utils.debug import SamplePoints, measure_memory, timeit
from proteinggnnmetrics.utils.functions import flatten_lists, tqdm_joblib


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf_2")
def main(cfg: DictConfig):
    os.makedirs(here() / cfg.experiments.results, exist_ok=True)
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    correlations = pd.DataFrame(columns=["epsilon", "pearson", "spearman"])

    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
        ),
        ("sample", SamplePoints(n=2)),
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
        pdb_files[
            cfg.experiments.proteins.not_perturbed.lower_bound : cfg.experiments.proteins.not_perturbed.upper_bound
            + 1
        ]
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
            pdb_files[
                cfg.experiments.proteins.perturbed.lower_bound : cfg.experiments.proteins.perturbed.upper_bound
                + 1
            ]
        )

        diagrams = [
            protein.descriptors["contact_graph"]["diagram"]
            for protein in proteins
        ]
        diagrams_perturbed = [
            protein.descriptors["contact_graph"]["diagram"]
            for protein in proteins_perturbed
        ]

        kernel = PersistenceFisherKernel(n_jobs=cfg.compute.n_jobs)
        res = kernel.compute_gram_matrix(
            np.array(diagrams), np.array(diagrams_perturbed)
        )
        mmd = MaximumMeanDiscrepancy(
            biased=True,
            squared=True,
            kernel=PersistenceFisherKernel(n_jobs=cfg.compute.n_jobs),  # type: ignore
        ).compute(diagrams, diagrams_perturbed)
        results.append({"mmd": mmd, "twist": twist})
    results = pd.DataFrame(results).to_csv(
        here() / cfg.experiments.results / "mmd_single_run_twist.csv"
    )


if __name__ == "__main__":
    main()
