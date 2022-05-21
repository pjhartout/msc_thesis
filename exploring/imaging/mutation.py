#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""muation.py

Introduce mutation and save resulting point cloud images.

"""

import hydra
import numpy as np
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.perturbations import (
    GaussianNoise,
    Mutation,
    Shear,
    Taper,
    Twist,
)
from proteinggnnmetrics.utils.debug import measure_memory, timeit


@hydra.main(config_path=str(here()) + "/conf/", config_name="config.yaml")
def main(cfg: DictConfig):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    for idx, i in tqdm(
        enumerate(np.arange(0, 1, 0.01)), total=len(np.arange(0, 1, 0.01)),
    ):
        base_feature_steps = [
            (
                "coordinates",
                Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
            ),
            (
                "mutate",
                Mutation(
                    p_mutate=i,
                    n_jobs=cfg.compute.n_jobs,
                    verbose=False,
                    random_state=np.random.RandomState(42),
                ),
            ),
        ]
        base_feature_pipeline = pipeline.Pipeline(
            base_feature_steps, verbose=False
        )
        proteins = base_feature_pipeline.fit_transform(pdb_files[:4])
        fig = proteins[3].plot_point_cloud(aa_as_color=True)
        camera = dict(eye=dict(x=1, y=2, z=1))
        fig.update_layout(scene_camera=camera, title="Mutate")

        fig.write_image(
            here() / cfg.imaging.mutation_path / f"protein_straight_{idx}.png",
            scale=5,
        )

    # curly protein
    # proteins[14].plot_point_cloud().show()
    # # straight protein
    # proteins[3].plot_point_cloud().show()
    # for i in range(len(proteins)):
    #     proteins[i].plot_point_cloud().show()


if __name__ == "__main__":
    main()
