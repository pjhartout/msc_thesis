#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""embeddings.py

Test out esm embeddings on a protein.
"""

import hydra
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinggnnmetrics.embeddings import ESM
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
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
            ("sequence", Sequence(n_jobs=cfg.compute.n_jobs),),
            ("esm", ESM(n_jobs=cfg.compute.n_jobs, verbose=cfg.verbose)),
        ]
    )
    proteins = feat_pipeline.fit_transform(pdb_files[:10])
    print("Done")


if __name__ == "__main__":
    main()
