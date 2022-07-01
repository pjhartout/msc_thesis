#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""illustrationss.py

Introduce gaussian noise to proteins and save resulting point cloud images.

"""

import hydra
import matplotlib.pyplot as plt
import plotly.graph_objs as gobj
import seaborn as sns
from gtda import pipeline
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.loaders import list_pdb_files
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates
from proteinmetrics.perturbations import GaussianNoise, Shear, Taper, Twist
from proteinmetrics.utils.debug import measure_memory, timeit
from proteinmetrics.utils.plots import setup_plotting_parameters

poi = 3


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    setup_plotting_parameters()
    base_feature_steps = [
        ("coordinates", Coordinates(granularity="CA", n_jobs=1),),
    ]
    base_feature_pipeline = pipeline.Pipeline(
        base_feature_steps, verbose=False
    )
    gn_steps = base_feature_steps.copy()
    gn_steps.append(
        (
            "gaussian noise",
            GaussianNoise(
                noise_mean=0,
                noise_std=3,
                n_jobs=1,
                verbose=False,
                random_state=42,
            ),
        ),  # type: ignore
    )
    gn_pipeline = pipeline.Pipeline(gn_steps, verbose=False)

    twist_steps = base_feature_steps.copy()
    twist_steps.append(
        (
            "twist",
            Twist(alpha=0.02, n_jobs=1, verbose=False, random_state=42,),
        ),  # type: ignore
    )
    twist_pipeline = pipeline.Pipeline(twist_steps, verbose=False)

    taper_steps = base_feature_steps.copy()
    taper_steps.append(
        (
            "taper",
            Taper(a=0.008, b=0.008, n_jobs=1, verbose=False, random_state=42,),
        ),  # type: ignore
    )
    taper_pipeline = pipeline.Pipeline(taper_steps, verbose=False)

    shear_steps = base_feature_steps.copy()
    shear_steps.append(
        (
            "shear",
            Shear(
                shear_x=1, shear_y=1, n_jobs=1, verbose=False, random_state=42,
            ),
        ),  # type: ignore
    )
    shear_pipeline = pipeline.Pipeline(shear_steps, verbose=False)

    protein = base_feature_pipeline.fit_transform([pdb_files[3]])
    protein_gauss = gn_pipeline.fit_transform([pdb_files[3]])
    protein_twist = twist_pipeline.fit_transform([pdb_files[3]])
    protein_taper = taper_pipeline.fit_transform([pdb_files[3]])
    protein_shear = shear_pipeline.fit_transform([pdb_files[3]])

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    fig = plt.figure(figsize=(10, 10))
    cmap = ListedColormap(sns.color_palette("mako_r", 256).as_hex())

    # =============
    # First subplot
    # =============
    # set up the axes for the first plot
    ax = fig.add_subplot(2, 2, 1, projection="3d")

    # plot a 3D surface like in the example mplot3d/surface3d_demo
    sc = ax.scatter(
        protein_gauss[0].coordinates[:, 0],
        protein_gauss[0].coordinates[:, 1],
        protein_gauss[0].coordinates[:, 2],
        s=40,
        c=range(len(protein_gauss[0].coordinates[:, 0])),
        marker="o",
        cmap=cmap,
        alpha=1,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-130, 130)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-150, 150)
    ax.title.set_text(r"Gaussian Noise ($\sigma=3\mathring{A}$)")

    # fig.colorbar(sc, shrink=0.5, aspect=10)

    # ==============
    # Second subplot
    # ==============
    # set up the axes for the second plot
    ax = fig.add_subplot(2, 2, 2, projection="3d")

    # plot a 3D surface like in the example mplot3d/surface3d_demo
    sc = ax.scatter(
        protein_twist[0].coordinates[:, 0],
        protein_twist[0].coordinates[:, 1],
        protein_twist[0].coordinates[:, 2],
        s=40,
        c=range(len(protein_twist[0].coordinates[:, 0])),
        marker="o",
        cmap=cmap,
        alpha=1,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-130, 130)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-150, 150)
    ax.title.set_text(r"Twist ($\alpha=0.02$ rad $\cdot \mathring{A}^{-1}$)")

    # fig.colorbar(sc, shrink=0.5, aspect=10)

    # ==============
    # Third subplot
    # ==============
    # set up the axes for the second plot
    ax = fig.add_subplot(2, 2, 3, projection="3d")

    # plot a 3D surface like in the example mplot3d/surface3d_demo
    sc = ax.scatter(
        protein_taper[0].coordinates[:, 0],
        protein_taper[0].coordinates[:, 1],
        protein_taper[0].coordinates[:, 2],
        s=40,
        c=range(len(protein_taper[0].coordinates[:, 0])),
        marker="o",
        cmap=cmap,
        alpha=1,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-130, 130)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-150, 150)
    ax.title.set_text(r"Taper ($a=b=0.08$)")

    # fig.colorbar(sc, shrink=0.5, aspect=10)

    # ==============
    # Fourth subplot
    # ==============
    # set up the axes for the second plot
    ax = fig.add_subplot(2, 2, 4, projection="3d")

    # plot a 3D surface like in the example mplot3d/surface3d_demo
    sc = ax.scatter(
        protein_shear[0].coordinates[:, 0],
        protein_shear[0].coordinates[:, 1],
        protein_shear[0].coordinates[:, 2],
        s=40,
        c=range(len(protein_shear[0].coordinates[:, 0])),
        marker="o",
        cmap=cmap,
        alpha=1,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-130, 130)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-150, 150)
    ax.title.set_text(r"Shear ($x=y=1$)")

    # fig.colorbar(sc, shrink=0.5, aspect=10)

    plt.savefig(
        here() / "exploring" / "illustrations" / "protein_perturbed.pdf",
        bbox_inches="tight",
    )

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    cmap = ListedColormap(sns.color_palette("mako_r", 256).as_hex())
    sc = ax.scatter(
        protein[0].coordinates[:, 0],
        protein[0].coordinates[:, 1],
        protein[0].coordinates[:, 2],
        s=40,
        c=range(len(protein[0].coordinates[:, 0])),
        marker="o",
        cmap=cmap,
        alpha=1,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-130, 130)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-150, 150)
    ax.title.set_text(r"Unperturbed")

    plt.savefig(
        here() / "exploring" / "illustrations" / "protein_unperturbed.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
