#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""select_protein.py

Select nice proteins to work with.

"""

import hydra
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here

from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.loaders import list_pdb_files
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates
from proteinmetrics.perturbations import Shear, Taper, Twist
from proteinmetrics.utils.debug import measure_memory, timeit


@hydra.main(config_path=str(here()) + "/conf/", config_name="config.yaml")
def main(cfg: DictConfig):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
        ),
    ]
    base_feature_pipeline = pipeline.Pipeline(
        base_feature_steps, verbose=False
    )
    proteins = base_feature_pipeline.fit_transform(pdb_files[:20])
    print(f"Protein name: {proteins[14].name}")
    # curly protein
    proteins[14].plot_point_cloud().show()
    # straight protein
    proteins[3].plot_point_cloud().show()
    # for i in range(len(proteins)):
    #     proteins[i].plot_point_cloud().show()


if __name__ == "__main__":
    main()
