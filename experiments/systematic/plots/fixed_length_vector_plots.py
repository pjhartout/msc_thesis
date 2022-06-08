#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""fixed_length_vector_plots.py

"""

import logging
import os
from pathlib import Path

import hydra
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here

from proteinmetrics.utils.functions import make_dir

log = logging.getLogger(__name__)

mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(
    fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf"
)
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False
# plt.rcParams["figure.figsize"] = (2, 2)
# plt.rcParams["savefig.dpi"] = 1200


def handle_perturbation(perturbation):
    if perturbation == "gaussian_noise":
        perturbation = "Gaussian Noise"
    elif perturbation == "mutation":
        perturbation = "Mutation"
    elif perturbation == "remove_edge":
        perturbation = "Remove Edge"
    elif perturbation == "add_edge":
        perturbation = "Add Edge"
    elif perturbation == "rewire_edge":
        perturbation = "Rewire Edge"
    elif perturbation == "twist":
        perturbation = "Twist"
    elif perturbation == "shear":
        perturbation = "Shear"
    elif perturbation == "taper":
        perturbation = "Taper"
    return perturbation


def handle_representation(graph_type):
    if graph_type == "pc_descriptor":
        graph_type = ""
    elif graph_type == "knn_graph":
        graph_type = r"$k$-NN Graph"
    elif graph_type == "eps_graph":
        graph_type = r"$\varepsilon$-Graph"
    return graph_type


def handle_descriptor(descriptor):
    if descriptor == "clustering_coefficient":
        descriptor = "Clustering Coefficient"
    elif descriptor == "degree_histogram":
        descriptor = "Degree Histogram"
    elif descriptor == "laplacian_spectrum_histogram":
        descriptor = "Laplacian Spectrum Histogram"
    elif descriptor == "dihedral_angle_histogram":
        descriptor = "Dihedral Angle Histogram"
    elif descriptor == "distance_histogram":
        descriptor = "Distance Histogram"
    return descriptor


def build_title(descriptor, perturbation, representation, extraction_param):
    if representation == "":
        title = f"descriptor = {descriptor}, perturbation = {perturbation}"
    else:
        if representation == r"$k$-NN Graph":
            title = (
                f"descriptor = {descriptor}, perturbation = {perturbation}, "
                + f"representation = {representation}, "
                + r"$\varepsilon$"
                + f" = {extraction_param}"
            )
        else:
            title = (
                f"descriptor = {descriptor}, perturbation = {perturbation}, "
                + f"representation = {representation}, "
                + r"$\varepsilon$"
                + f" = {extraction_param}"
            )
    return title.lower()


def build_plot(cfg, path: Path) -> None:
    df = pd.read_csv(path)
    gauss_kernel_cols = [
        col for col in df.columns if "sigma" in col or "linear" in col
    ]
    for col in gauss_kernel_cols:
        # Normalize
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    df = df.melt(
        id_vars=["run", "perturb"], var_name="kernel", value_name="mmd"
    )
    df.rename(columns={"perturb": r"Std. ($\AA$)"}, inplace=True)

    parsed_path = str(path).split("/")
    descriptor = handle_descriptor(parsed_path[-2])
    perturbation = handle_perturbation(parsed_path[-3])
    extraction_param = parsed_path[-4]
    representation = handle_representation(parsed_path[-5])

    # Clean up kernel names
    # df.kernel = df.kernel.str.replace("linear_kernel", "Linear Kernel")
    # df.kernel = df.kernel.str.replace("sigma=", "Gaussian kernel sigma =")
    df = df.rename(columns={"mmd": "MMD",},)
    # Initialize a grid of plots with an Axes for each kernel config
    palette = sns.color_palette("mako_r", df.kernel.nunique())
    g = sns.relplot(
        data=df,
        x=r"Std. ($\AA$)",
        y="MMD",
        col="kernel",
        hue="kernel",
        kind="line",
        col_wrap=4,
        height=2.7,
        aspect=0.8,
        palette=palette,
        ci=100,
    )
    g.fig.suptitle(
        build_title(
            descriptor, perturbation, representation, extraction_param
        ),
        fontsize=16,
    )

    plt.legend([], [], frameon=False)
    g.legend.remove()
    titles = [
        1.0e-05,
        1.0e-04,
        1.0e-03,
        1.0e-02,
        1.0e-01,
        1,
        1.0e02,
        1.0e03,
        1.0e04,
        1.0e05,
        0,
    ]
    for i, ax in enumerate(g.axes.flatten()):
        if titles[i] != 0:
            ax.set_title(r"RBF Kernel $\sigma$ " + f" = {titles[i]}")
        else:
            ax.set_title(f"Linear Kernel")
    # plt.title("Test")
    plt.tight_layout()
    # Adjust the arrangement of the plots
    # plt.savefig("relplot_test.svg")
    make_dir(cfg.imaging.systematic_plots)
    plt.savefig(
        here()
        / cfg.imaging.systematic_plots
        / f"relplot_{descriptor.lower()}_{perturbation.lower()}_{representation.lower()}_{extraction_param.lower()}.png"
    )


@hydra.main(
    version_base=None,
    config_path=str(here()) + "/conf/",
    config_name="systematic",
)
def main(cfg: DictConfig):
    for root, dirs, files in os.walk(
        here() / "data/systematic/human/fixed_length_kernels/", topdown=False
    ):
        if files != [] and ".csv" in files[0]:
            build_plot(cfg, Path(root + "/" + files[0]))
    # path = (
    #     here()
    #     / "data/systematic/human/fixed_length_kernels/pc_descriptor/1/gaussian_noise/distance_histogram/gaussian_noise_mmds.csv"
    # )
    # build_plot(cfg, path)


if __name__ == "__main__":
    main()
