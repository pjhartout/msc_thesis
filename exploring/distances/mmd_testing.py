#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""mmd_testing.py

Computes MMD test between two distributions

"""


from tabnanny import verbose

import hydra
import numpy as np
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here

from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.loaders import list_pdb_files, load_graphs
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.statistics import MMDTest


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
    ]
    base_feature_pipeline = pipeline.Pipeline(
        base_feature_steps, verbose=False
    )
    proteins = base_feature_pipeline.fit_transform(pdb_files[:240])
    dist_1 = load_graphs(proteins[:120], "eps_graph")
    dist_2 = load_graphs(proteins[120:], "eps_graph")

    p_value = MMDTest(
        alpha=0.05, m=100, t=100, kernel=WeisfeilerLehmanKernel(n_jobs=cfg.compute.n_jobs), verbose=True  # type: ignore
    ).compute_p_value(dist_1, dist_2)
    print(f"p-value: {p_value}")


if __name__ == "__main__":
    main()
