#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""embeddings.py

Test out esm embeddings on a protein.
"""

import hydra
import numpy as np
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.embeddings import ESM
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.kernels import LinearKernel
from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates, Sequence
from proteinggnnmetrics.perturbations import GaussianNoise, Shear, Taper, Twist
from proteinggnnmetrics.utils.debug import measure_memory, timeit


@hydra.main(config_path=str(here()) + "/conf/", config_name="config.yaml")
def main(cfg: DictConfig):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    feat_pipeline = pipeline.Pipeline(
        [
            (
                "sequence",
                Sequence(n_jobs=cfg.compute.n_jobs),
            ),
            (
                "esm",
                ESM(size="M", n_jobs=cfg.compute.n_jobs, verbose=cfg.verbose),
            ),
        ]
    )
    proteins = feat_pipeline.fit_transform(pdb_files[:7])

    reps_1 = np.array([protein.embeddings["esm"] for protein in proteins[:4]])
    reps_2 = np.array([protein.embeddings["esm"] for protein in proteins[4:]])

    # Compute MMD
    mmd = MaximumMeanDiscrepancy(
        kernel=LinearKernel(dense_output=False, normalize=False),
        biased=True,
        squared=True,
        verbose=cfg.verbose,
    )
    mmd_value = mmd.compute(reps_1, reps_2)
    print(f"MMD: {mmd_value}")


if __name__ == "__main__":
    main()
