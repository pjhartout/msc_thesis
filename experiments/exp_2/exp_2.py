#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""exp_2.py

The goal of this experiment is to investigate the effect of twisting on MMD features derived from TDA.

"""

import os

import hydra
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
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
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.functions import flatten_lists, tqdm_joblib


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf")
def main(cfg: DictConfig):
    os.makedirs(here() / cfg.experiments.results, exist_ok=True)
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    correlations = pd.DataFrame(columns=["epsilon", "pearson", "spearman"])

    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
        ),
        ("contact map", ContactMap(n_jobs=cfg.compute.n_jobs,),),
        (
            "tda",
            TopologicalDescriptor(
                "diagram",
                epsilon=0.01,
                n_bins=100,
                order=2,
                n_jobs=cfg.compute.n_jobs,
                landscape_layers=1,
            ),
        ),
    ]

    base_feature_pipeline = pipeline.Pipeline(
        base_feature_steps, verbose=False
    )
    print("Fit unperturbed")
    proteins = base_feature_pipeline.fit_transform(
        pdb_files[
            cfg.experiments.proteins.not_perturbed.lower_bound : cfg.experiments.proteins.not_perturbed.upper_bound
            + 1
        ]
    )

    for twist in tqdm(
        np.arange(
            cfg.perturbations.twist.min,
            cfg.perturbations.twist.max,
            cfg.perturbations.twist.step,
        ),
        position=1,
        leave=False,
        desc="Twist range",
    ):
        perturb_feature_steps = flatten_lists(
            [
                base_feature_steps[:2]
                + [
                    (
                        "twist",
                        Twist(
                            alpha=twist,
                            random_state=42,
                            n_jobs=cfg.compute.n_jobs,
                            verbose=False,
                        ),
                    )
                ]
                + base_feature_steps[2:]
            ]
        )
        perturb_feature_pipeline = pipeline.Pipeline(
            base_feature_steps, verbose=False
        )
        proteins_perturbed = perturb_feature_pipeline.fit_transform(
            pdb_files[
                cfg.eps_var.proteins.perturbed.lower_bound : cfg.eps_var.proteins.perturbed.upper_bound
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
        # mmd = MaximumMeanDiscrepancy(
        #     biased=True,
        #     squared=True,
        #     kernel=PersistenceFisherKernel(),  # type: ignore
        # ).compute(graphs, graphs_perturbed)
        # results.append({"mmd": mmd, "std": std})


if __name__ == "__main__":
    main()
