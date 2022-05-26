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

from proteinmetrics.descriptors import ESM
from proteinmetrics.distance import MaximumMeanDiscrepancy
from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.kernels import LinearKernel
from proteinmetrics.loaders import list_pdb_files
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates, Sequence
from proteinmetrics.perturbations import GaussianNoise, Shear, Taper, Twist
from proteinmetrics.utils.debug import measure_memory, timeit


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf")
def main(cfg: DictConfig):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    proteins = Sequence(n_jobs=cfg.compute.n_jobs).fit_transform(pdb_files[:7])
    proteins = ESM(
        size="M",
        n_jobs=cfg.compute.n_jobs,
        verbose=cfg.debug.verbose,
        longest_sequence=max([len(protein.sequence) for protein in proteins]),
    ).fit_transform(proteins)

    reps_1 = np.array([protein.embeddings["esm"] for protein in proteins[:4]])
    reps_2 = np.array([protein.embeddings["esm"] for protein in proteins[4:]])

    # Compute MMD
    mmd = MaximumMeanDiscrepancy(
        kernel=LinearKernel(
            dense_output=False,
            normalize=False,
            n_jobs=cfg.compute.n_jobs,
            verbose=cfg.debug.verbose,
        ),
        biased=True,
        squared=True,
        verbose=cfg.debug.verbose,
    )
    mmd_value = mmd.compute(reps_1, reps_2)
    print(f"MMD: {mmd_value}")


if __name__ == "__main__":
    main()
