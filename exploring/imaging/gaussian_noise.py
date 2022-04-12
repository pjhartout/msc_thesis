#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""gaussian_noise.py

Introduce gaussian noise to proteins and save resulting point cloud images.

"""

import hydra
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.perturbations import GaussianNoise, Shear, Taper, Twist
from proteinggnnmetrics.utils.debug import measure_memory, timeit


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf")
def main(cfg: DictConfig):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    for i in tqdm(range(0, 50)):
        base_feature_steps = [
            (
                "coordinates",
                Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
            ),
            (
                "gaussian noise",
                GaussianNoise(
                    noise_mean=0,
                    noise_variance=i,
                    n_jobs=cfg.compute.n_jobs,
                    verbose=False,
                    random_state=42,
                ),
            ),
        ]
        base_feature_pipeline = pipeline.Pipeline(
            base_feature_steps, verbose=False
        )
        proteins = base_feature_pipeline.fit_transform(pdb_files[:4])
        fig = proteins[cfg.imaging.protein_of_interest].plot_point_cloud()
        camera = dict(eye=dict(x=1, y=2, z=1))
        fig.update_layout(scene_camera=camera, title="Gaussian noise")
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=[-120, 100],
                ),
                yaxis=dict(
                    range=[-60, 60],
                ),
                zaxis=dict(
                    range=[-120, 120],
                ),
                aspectmode="manual",
                aspectratio=dict(x=1, y=0.5, z=1),
            ),
        )
        fig.write_image(
            here()
            / cfg.imaging.gaussian_noise_path
            / f"protein_straight_{i}.png",
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
