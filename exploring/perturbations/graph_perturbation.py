#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""perturbation_exploration.py

Making some experiments with Gaussian noise
"""

import os
import random
from datetime import datetime

import hydra
import numpy as np
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here

from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.loaders import list_pdb_files
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates
from proteinmetrics.perturbations import Mutation

config = configure()

N_JOBS = 10
REDUCE_DATA = False


@hydra.main(config_path=str(here()) + "/conf", config_name="config.yaml")
def main(cfg: DictConfig):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    base_feature_steps = [
        ("coordinates", Coordinates(granularity="CA", n_jobs=N_JOBS)),
        ("contact map", ContactMap(metric="euclidean", n_jobs=N_JOBS)),
        ("epsilon graph", EpsilonGraph(epsilon=4, n_jobs=N_JOBS)),
        (
            "rewire",
            Mutation(
                p_mutate=0.1,
                n_jobs=N_JOBS,
                graph_type="eps_graph",
                random_state=np.random.RandomState(42),
                verbose=True,
            ),
        ),
    ]

    base_feature_pipeline = pipeline.Pipeline(
        base_feature_steps, verbose=False
    )
    proteins = base_feature_pipeline.fit_transform(pdb_files[:100])


if __name__ == "__main__":
    main()
