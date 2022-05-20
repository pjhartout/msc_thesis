#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""computation_budget_estimator.py

This script helps generate an estimate of the execution time of running the whole pipeline.

"""

import os
import random

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pyprojroot import here


@hydra.main(
    version_base=None, config_path=str(here()) + "/conf/", config_name="conf"
)
def main(cfg):
    experiments = pd.read_csv(
        here() / "data" / "systematic" / "experimental_configurations.csv",
        index_col=0,
    )
    descriptors = pd.read_csv(
        here()
        / cfg.meta.data.time_estimates_dir
        / "descriptor_benchmarks.csv",
        index_col=0,
    )
    kernels = pd.read_csv(
        here() / cfg.meta.data.time_estimates_dir / "kernel_benchmarks.csv",
        index_col=0,
    )
    representations = pd.read_csv(
        here()
        / cfg.meta.data.time_estimates_dir
        / "representation_benchmarks.csv",
        index_col=0,
    )
    budgets = list()
    for idx, experiment in experiments.iterrows():
        budget = 0

        if experiment.descriptors == "esm":
            budget = representations.loc["esm"].values[0]
        elif experiment.descriptors == "persistence_diagram":
            budget = representations.loc["tda"].values[0]
        elif "graph" in experiment.representations:
            budget = representations.loc["graph"].values[0]

        if experiment.descriptors in [
            "degree_histogram",
            "laplacian_spectrum",
            "clustering_histogram",
            "angles",
        ]:
            budget += descriptors.loc["degree"].values[0]
        elif experiment.descriptors == "esm":
            budget += 0
        elif experiment.descriptors == "persistence_diagram":
            budget += representations.loc["tda"].values[0]

        if experiment.kernel == "linear":
            budget += kernels.loc["linear"].values[0]
        elif experiment.kernel == "gaussian":
            budget += kernels.loc["gaussian"].values[0]
        elif experiment.kernel == "weisfeiler-lehman":
            budget += kernels.loc["wl"].values[0]
        elif experiment.kernel == "persistence_fisher":
            budget += kernels.loc["pf"].values[0]

        budgets.append(budget)

    experiments["budget"] = budgets
    N_SAMPLES = 100
    N_PERTURB = 50
    experiments["budget"] = experiments["budget"] * N_PERTURB * N_SAMPLES

    experiments.to_csv(
        here()
        / "data"
        / "systematic"
        / "experimental_configurations_w_budget.csv"
    )


if __name__ == "__main__":
    main()
