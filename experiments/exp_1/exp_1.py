#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""exp_2.py

The goal of this experiment is to see how the choice of epsilon affects the
behaviour of MMD.

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
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinggnnmetrics.distance import (
    MaximumMeanDiscrepancy,
    PearsonCorrelation,
    SpearmanCorrelation,
)
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.loaders import list_pdb_files, load_graphs
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.perturbations import GaussianNoise
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.functions import flatten_lists, tqdm_joblib


def execute_run(cfg, run):
    os.makedirs(here() / cfg.eps_var.paths.results / str(run), exist_ok=True)
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    correlations = pd.DataFrame(columns=["epsilon", "pearson", "spearman"])
    for epsilon in tqdm(
        range(
            cfg.eps_var.eps.lower_bound,
            cfg.eps_var.eps.upper_bound,
            cfg.eps_var.eps.step,
        ),
        position=1,
        leave=False,
        desc="Epsilon range",
    ):
        base_feature_steps = [
            (
                "coordinates",
                Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
            ),
            (
                "contact map",
                ContactMap(metric="euclidean", n_jobs=cfg.compute.n_jobs,),
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
            desc="STD range",
        ):
            perturb_feature_steps = flatten_lists(
                [
                    base_feature_steps[:1]
                    + [
                        (
                            "gauss",
                            GaussianNoise(
                                random_seed=42,
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

        results.to_csv(here() / cfg.eps_var.paths.results / str(run) / f"epsilon_{epsilon}.csv")  # type: ignore
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
    correlations.to_csv(
        here() / cfg.eps_var.paths.results / str(run) / f"correlations.csv"
    )


@timeit
@measure_memory
@hydra.main(config_path=str(here()) + "/conf/", config_name="conf")
def main(cfg: DictConfig):
    os.makedirs(here() / cfg.eps_var.paths.results, exist_ok=True)
    with tqdm_joblib(
        tqdm(
            desc="Execute n_runs",
            total=len(list(range(cfg.compute.n_parallel_runs))),
        )
    ) as progressbar:
        Parallel(n_jobs=cfg.compute.n_parallel_runs)(
            delayed(execute_run)(cfg, run) for run in range(cfg.eps_var.n_runs)
        )


if __name__ == "__main__":
    main()
