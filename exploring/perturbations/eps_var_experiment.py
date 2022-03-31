#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""eps_var.py

The goal of this experiment is to see how the choice of epsilon affects the
behaviour of MMD.

"""

import os
import random
from datetime import datetime

import hydra
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinggnnmetrics.descriptors import DegreeHistogram
from proteinggnnmetrics.distance import (
    MaximumMeanDiscrepancy,
    PearsonCorrelation,
    SpearmanCorrelation,
)
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph, KNNGraph
from proteinggnnmetrics.kernels import LinearKernel
from proteinggnnmetrics.loaders import (
    list_pdb_files,
    load_descriptor,
    load_graphs,
)
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.perturbations import GaussianNoise
from proteinggnnmetrics.protein import Protein
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.functions import flatten_lists


@timeit
@measure_memory
@hydra.main(config_path=str(here()) + "conf/", config_name="config.yaml")
def main(cfg: DictConfig):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(CACHE_DIR / f"{now}", exist_ok=True)
    OUT_DIR = CACHE_DIR / f"{now}"
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    for run in tqdm(
        range(cfg.eps_var.compute.n_runs), position=0, leave=False
    ):
        correlations = pd.DataFrame(columns=["epsilon", "pearson", "spearman"])
        for epsilon in tqdm(
            range(
                cfg.eps_var.eps.lower_bound,
                cfg.eps_var.eps.upper_bound,
                cfg.eps_var.eps.step,
            ),
            position=1,
            leave=False,
        ):
            base_feature_steps = [
                (
                    "coordinates",
                    Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
                ),
                (
                    "contact map",
                    ContactMap(
                        metric="euclidean",
                        n_jobs=cfg.compute.n_jobs,
                    ),
                ),
                (
                    "epsilon graph",
                    EpsilonGraph(epsilon=epsilon, n_jobs=cfg.compute.n_jobs),
                ),
            ]

            base_feature_pipeline = pipeline.Pipeline(
                base_feature_steps, verbose=False
            )
            proteins = base_feature_pipeline.fit_transform(
                pdb_files[
                    cfg.eps_var.proteins.not_perturbed.lower_bound : cfg.eps_var.proteins.not_perturbed.upper_bound
                ]
            )
            results = list()
            for std in tqdm(
                np.arange(
                    cfg.eps_var.std.lower_bound,
                    cfg.eps_var.std.upper_bound,
                    cfg.eps_var.std.step,
                ),
                position=2,
                leave=False,
            ):
                perturb_feature_steps = flatten_lists(
                    [
                        base_feature_steps[:1]
                        + [
                            (
                                "gauss",
                                GaussianNoise(
                                    random_state=42,
                                    noise_mean=0,
                                    noise_variance=std,
                                    n_jobs=cfg.compute.n_jobs,
                                    verbose=False,
                                ),
                            )
                        ]
                        + base_feature_steps[1:]
                    ]
                )

                perturb_feature_pipeline = pipeline.Pipeline(
                    perturb_feature_steps, verbose=False
                )
                proteins_perturbed = perturb_feature_pipeline.fit_transform(
                    pdb_files[
                        cfg.eps_var.proteins.perturbed.lower_bound : cfg.eps_var.proteins.perturbed.upper_bound
                    ]
                )
                graphs = load_graphs(proteins, graph_type="eps_graph")
                graphs_perturbed = load_graphs(
                    proteins_perturbed, graph_type="eps_graph"
                )
                mmd = MaximumMeanDiscrepancy(
                    biased=True,
                    squared=True,
                    kernel=WeisfeilerLehmanKernel(
                        n_jobs=cfg.compute.n_jobs,
                        n_iter=5,
                        normalize=True,
                        biased=True,
                    ),  # type: ignore
                ).compute(graphs, graphs_perturbed)
                results.append({"mmd": mmd, "std": std})
                # print(f"{mmd:.2f}")

            # Convert mmd and params to dataframe
            results = pd.DataFrame(data=results)

            results.to_csv(OUT_DIR / f"results_epsilon_{epsilon}_{now}.csv")  # type: ignore
            spearman_correlation = SpearmanCorrelation().compute(
                results["mmd"].values, results["std"].values  # type: ignore
            )
            pearson_correlation = PearsonCorrelation().compute(
                results["mmd"].values, results["std"].values  # type: ignore
            )
            correlations = pd.concat(
                [
                    correlations,
                    pd.DataFrame(
                        {
                            "epsilon": [epsilon],
                            "pearson": [pearson_correlation],
                            "spearman": [spearman_correlation],
                        }
                    ),
                ]
            )
        correlations.to_csv(OUT_DIR / f"correlations_{run}.csv")


if __name__ == "__main__":
    main()
