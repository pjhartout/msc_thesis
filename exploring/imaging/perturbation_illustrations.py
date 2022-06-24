#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""gaussian_noise.py

Introduce gaussian noise to proteins and save resulting point cloud images.

"""

import hydra
import plotly.graph_objs as gobj
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.loaders import list_pdb_files
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates
from proteinmetrics.perturbations import GaussianNoise, Shear, Taper, Twist
from proteinmetrics.utils.debug import measure_memory, timeit


def plot_pc(protein):
    scene = {
        "xaxis": {
            "title": "x",
            "type": "linear",
            "showexponent": "all",
            "exponentformat": "e",
        },
        "yaxis": {
            "title": "y",
            "type": "linear",
            "showexponent": "all",
            "exponentformat": "e",
        },
        "zaxis": {
            "title": "z",
            "type": "linear",
            "showexponent": "all",
            "exponentformat": "e",
        },
    }

    fig = gobj.Figure()
    fig.update_layout(scene=scene)

    fig.add_trace(
        gobj.Scatter3d(
            x=protein.coordinates[:, 0],
            y=protein.coordinates[:, 1],
            z=protein.coordinates[:, 2],
            mode="markers",
            marker={
                "size": 4,
                "color": list(range(protein.coordinates.shape[0])),
                "colorscale": "Viridis",
                "opacity": 0.8,
            },
        )
    )

    return fig


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf")
def main(cfg: DictConfig):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
        ),
        (
            "gaussian noise",
            GaussianNoise(
                noise_mean=0,
                noise_std=4,
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
    fig = plot_pc(proteins[cfg.imaging.protein_of_interest])
    camera = dict(eye=dict(x=1, y=2, z=1))
    fig.update_layout(scene_camera=camera, title="Gaussian noise")
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-120, 100],),
            yaxis=dict(range=[-60, 60],),
            zaxis=dict(range=[-120, 120],),
            aspectmode="manual",
            aspectratio=dict(x=1, y=0.5, z=1),
        ),
    )
    fig.write_image(
        here() / f"protein_straight_gaussian.png", scale=5,
    )


if __name__ == "__main__":
    main()
