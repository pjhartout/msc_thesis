#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""baselines.py

The idea here is to make a violin plot that shows the distribution of the
baseline MMD values.

"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from pyprojroot import here

from proteinmetrics.utils.plots import setup_plotting_parameters


def load_weisfeiler_lehman_baseline_data():
    df_wl_8 = pd.read_csv(
        here()
        / "data"
        / "systematic"
        / "human"
        / "weisfeiler_lehman"
        / "eps_graph"
        / "8"
        / "add_edges"
        / "add_edges_mmds.csv",
    )
    df_wl_8.assign(
        eps=8, kernel="weisfeiler_lehman",
    )

    df_wl_32 = pd.read_csv(
        here()
        / "data"
        / "systematic"
        / "human"
        / "weisfeiler_lehman"
        / "eps_graph"
        / "32"
        / "add_edges"
        / "add_edges_mmds.csv",
    )
    df_wl_32.assign(
        eps=32, kernel="weisfeiler_lehman",
    )
    return pd.concat([df_wl_8, df_wl_32])


def load_tda_baseline_data():
    df = pd.DataFrame()
    for i in range(10):
        df_run = pd.read_csv(
            here()
            / "data"
            / "systematic"
            / "human"
            / "tda"
            / "gaussian_noise"
            / str(i)
            / "gaussian_noise_mmds.csv",
        )
        df_run.assign(kernel="tda", run=i)
        df = pd.concat([df, df_run])
    return df


def load_esm_baseline_data():
    df = pd.DataFrame()
    for i in range(10):
        df_run = pd.read_csv(
            here()
            / "data"
            / "systematic"
            / "human"
            / "fixed_length_kernels"
            / "esm"
            / "mutation"
            / str(i)
            / "mutation_mmds.csv",
        )
        df_run.assign(kernel="esm", run=i)
        df = pd.concat([df, df_run])
    return df


def load_baseline_clustering_data():
    df_8_clustering = pd.read_csv(
        here()
        / "data"
        / "systematic"
        / "human"
        / "fixed_length_kernels"
        / "eps_graph"
        / "8"
        / "add_edges"
        / "clustering_histogram"
        / "add_edges_mmds.csv",
    )
    df_8_clustering = df_8_clustering.assign(
        eps=8, kernel="clustering_histogram",
    )
    df_32_clustering = pd.read_csv(
        here()
        / "data"
        / "systematic"
        / "human"
        / "fixed_length_kernels"
        / "eps_graph"
        / "32"
        / "gaussian_noise"
        / "clustering_histogram"
        / "gaussian_noise_mmds.csv",
    )
    df_32_clustering = df_32_clustering.assign(
        eps=32, kernel="clustering_histogram",
    )
    return pd.concat([df_8_clustering, df_32_clustering])


def load_baseline_degree_data():
    df_8_degree = pd.read_csv(
        here()
        / "data"
        / "systematic"
        / "human"
        / "fixed_length_kernels"
        / "eps_graph"
        / "8"
        / "add_edges"
        / "degree_histogram"
        / "add_edges_mmds.csv",
    )
    df_8_degree = df_8_degree.assign(eps=8, kernel="degree_histogram",)
    df_32_degree = pd.read_csv(
        here()
        / "data"
        / "systematic"
        / "human"
        / "fixed_length_kernels"
        / "eps_graph"
        / "32"
        / "add_edges"
        / "degree_histogram"
        / "add_edges_mmds.csv",
    )
    df_32_degree = df_32_degree.assign(eps=32, kernel="degree_histogram",)
    return pd.concat([df_8_degree, df_32_degree])


def load_baseline_laplacian_spectrum_data():
    df_8_laplacian_spectrum = pd.read_csv(
        here()
        / "data"
        / "systematic"
        / "human"
        / "fixed_length_kernels"
        / "eps_graph"
        / "8"
        / "add_edges"
        / "laplacian_spectrum_histogram"
        / "add_edges_mmds.csv",
    )
    df_8_laplacian_spectrum = df_8_laplacian_spectrum.assign(
        eps=8, kernel="laplacian_spectrum_histogram",
    )
    df_32_laplacian_spectrum = pd.read_csv(
        here()
        / "data"
        / "systematic"
        / "human"
        / "fixed_length_kernels"
        / "eps_graph"
        / "32"
        / "gaussian_noise"
        / "laplacian_spectrum_histogram"
        / "gaussian_noise_mmds.csv",
    )
    df_32_laplacian_spectrum = df_32_laplacian_spectrum.assign(
        eps=32, kernel="laplacian_spectrum_histogram",
    )
    return pd.concat([df_8_laplacian_spectrum, df_32_laplacian_spectrum])


def plot_wl_baseline(df_wl):
    df_wl = df_wl.melt(id_vars=["run", "perturb"])
    df_wl = df_wl.rename(columns={"variable": "Iterations", "value": "MMDs"})

    df_wl["Iterations"] = df_wl["Iterations"].str.replace("n_iter=", "")

    palette = sns.color_palette("mako_r", df_wl.Iterations.nunique())

    g = sns.violinplot(x="Iterations", y="MMDs", data=df_wl, palette=palette)

    plt.savefig(here() / "exploring" / "baselines" / "wl_baselines.pdf",)


def plot_tda_baseline(df_tda):
    df_tda = df_tda.rename(
        columns={
            "persistence_fisher_bandwidth=1;bandwidth_fisher=1": "PFK",
            "mutli_scale_kernel_bandwidth=1;bandwidth_fisher=1": "MSK",
        }
    )
    df_tda = df_tda.melt(id_vars=["run", "perturb"])
    df_tda = df_tda.rename(columns={"value": "MMDs", "variable": "Kernel",})

    # df_esm["Iterations"] = df_esm["Iterations"].str.replace("n_iter=", "")

    palette = sns.color_palette("mako_r", df_tda.Kernel.nunique())

    g = sns.FacetGrid(
        df_tda,
        col="Kernel",
        hue="Kernel",
        col_wrap=2,
        sharey=False,
        palette=palette,
    )
    g.map(
        sns.violinplot,
        "Kernel",
        "MMDs",
        # x="Kernel",
        # # hue="eps",
        # palette=palette,
        split=True,
    )
    g.set(xticklabels=[])
    g.set(xlabel=None)
    titles = [
        "PFK",
        "MSK",
    ]
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title(titles[i])

    # g = sns.violinplot(x="Kernel", y="MMDs", data=df_esm, palette=palette)
    g.tight_layout()
    plt.savefig(here() / "exploring" / "baselines" / "tda_baselines.pdf",)


def plot_esm_baseline(df_esm):

    df_esm = df_esm.melt(id_vars=["run", "perturb"])
    df_esm = df_esm.rename(columns={"value": "MMDs", "variable": "Kernel",})
    palette = sns.color_palette("mako_r", df_esm.Kernel.nunique())

    g = sns.FacetGrid(
        df_esm,
        col="Kernel",
        hue="Kernel",
        col_wrap=3,
        sharey=False,
        palette=palette,
    )
    g.map(
        sns.violinplot,
        "Kernel",
        "MMDs",
        # x="Kernel",
        # # hue="eps",
        # palette=palette,
        split=True,
    )
    # g.set(xticklabels=[])
    # g.set(xlabel=None)
    # titles = [
    #     "PFK",
    #     "MSK",
    # ]
    # for i, ax in enumerate(g.axes.flatten()):
    #     ax.set_title(titles[i])

    # g = sns.violinplot(x="Kernel", y="MMDs", data=df_esm, palette=palette)
    g.tight_layout()
    plt.savefig(here() / "exploring" / "baselines" / "esm_baselines.pdf",)


def plot_clustering_baseline(df_clustering):
    df_clustering = df_clustering.melt(
        id_vars=["run", "perturb", "eps", "kernel"]
    )
    df_clustering = df_clustering.rename(
        columns={"value": "MMDs", "variable": "Kernel", "kernel": "descriptor"}
    )
    palette = sns.color_palette("mako_r", df_clustering.eps.nunique())

    g = sns.FacetGrid(df_clustering, col="Kernel", col_wrap=3, sharey=False,)
    g.map_dataframe(
        sns.violinplot,
        x="Kernel",
        y="MMDs",
        hue="eps",
        palette=palette,
        split=True,
    )
    # g.set(xticklabels=[])
    # g.set(xlabel=None)
    # titles = [
    #     "PFK",
    #     "MSK",
    # ]
    # for i, ax in enumerate(g.axes.flatten()):
    #     ax.set_title(titles[i])

    # g = sns.violinplot(x="Kernel", y="MMDs", data=df_esm, palette=palette)
    plt.tight_layout()
    plt.savefig(
        here() / "exploring" / "baselines" / "clustering_baselines.pdf",
    )


def plot_degree_baseline(df_degree):
    df_degree = df_degree.melt(id_vars=["run", "perturb", "eps", "kernel"])
    df_degree = df_degree.rename(
        columns={"value": "MMDs", "variable": "Kernel", "kernel": "descriptor"}
    )
    palette = sns.color_palette("mako_r", df_degree.eps.nunique())

    g = sns.FacetGrid(df_degree, col="Kernel", col_wrap=3, sharey=False,)
    g.map_dataframe(
        sns.violinplot,
        x="Kernel",
        y="MMDs",
        hue="eps",
        palette=palette,
        split=True,
    )
    # g.set(xticklabels=[])
    # g.set(xlabel=None)
    # titles = [
    #     "PFK",
    #     "MSK",
    # ]
    # for i, ax in enumerate(g.axes.flatten()):
    #     ax.set_title(titles[i])

    # g = sns.violinplot(x="Kernel", y="MMDs", data=df_esm, palette=palette)
    plt.tight_layout()
    plt.savefig(here() / "exploring" / "baselines" / "degree_baselines.pdf",)


def plot_laplacian_spectrum_baseline(df_laplacian_spectrum):
    df_laplacian_spectrum = df_laplacian_spectrum.melt(
        id_vars=["run", "perturb", "eps", "kernel"]
    )
    df_laplacian_spectrum = df_laplacian_spectrum.rename(
        columns={"value": "MMDs", "variable": "Kernel", "kernel": "descriptor"}
    )
    palette = sns.color_palette("mako_r", df_laplacian_spectrum.eps.nunique())

    g = sns.FacetGrid(
        df_laplacian_spectrum, col="Kernel", col_wrap=3, sharey=False,
    )
    g.map_dataframe(
        sns.violinplot,
        x="Kernel",
        y="MMDs",
        hue="eps",
        palette=palette,
        split=True,
    )
    # g.set(xticklabels=[])
    # g.set(xlabel=None)
    # titles = [
    #     "PFK",
    #     "MSK",
    # ]
    # for i, ax in enumerate(g.axes.flatten()):
    #     ax.set_title(titles[i])

    # g = sns.violinplot(x="Kernel", y="MMDs", data=df_esm, palette=palette)
    plt.tight_layout()
    plt.savefig(
        here()
        / "exploring"
        / "baselines"
        / "laplacian_spectrum_baselines.pdf",
    )


def main():
    setup_plotting_parameters()
    df_wl = load_weisfeiler_lehman_baseline_data()
    df_wl = df_wl.loc[df_wl.perturb == 0]
    df_tda = load_tda_baseline_data()
    df_tda = df_tda.loc[df_tda.perturb == 0]
    df_esm = load_esm_baseline_data()
    df_esm.loc[df_esm.perturb == 0]
    df_clustering = load_baseline_clustering_data()
    df_clustering = df_clustering.loc[df_clustering.perturb == 0]
    df_degree = load_baseline_degree_data()
    df_degree = df_degree.loc[df_degree.perturb == 0]
    df_laplacian_spectrum = load_baseline_laplacian_spectrum_data()
    df_laplacian_spectrum = df_laplacian_spectrum.loc[
        df_laplacian_spectrum.perturb == 0
    ]

    # plot_wl_baseline(df_wl)
    # plot_tda_baseline(df_esm)
    # plot_esm_baseline(df_esm)
    # plot_clustering_baseline(df_clustering)
    # plot_degree_baseline(df_degree)
    plot_laplacian_spectrum_baseline(df_laplacian_spectrum)


if __name__ == "__main__":
    main()
