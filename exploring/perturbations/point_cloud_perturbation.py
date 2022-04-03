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

from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.perturbations import Shear, Taper, Twist
from proteinggnnmetrics.utils.debug import measure_memory, timeit


@hydra.main(config_path=str(here()) + "/conf/", config_name="config.yaml")
def main(cfg: DictConfig):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    for i in np.arange(0, 0.2, 0.01):
        base_feature_steps = [
            (
                "coordinates",
                Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
            ),
            (
                "twist",
                Twist(
                    alpha=i,
                    random_state=cfg.random.state,
                    n_jobs=cfg.compute.n_jobs,
                    verbose=True,
                ),
            ),
        ]
        base_feature_pipeline = pipeline.Pipeline(
            base_feature_steps, verbose=False
        )
        proteins = base_feature_pipeline.fit_transform(pdb_files[:20])
        proteins[0].plot_point_cloud().show()


if __name__ == "__main__":
    main()
