#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""overall_mmd_behaviour_eps.py

Surprisingly, structured graphs show a better correlation with the MMD than unstructured graphs.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from pyprojroot import here

from proteinmetrics.utils.plots import setup_plotting_parameters

relevant_cols = ["perturb", "run", "sigma=0.01"]


def normalize(df):
    cols = [col for col in df.columns if "run" not in col]
    for col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def load_clustering() -> pd.DataFrame:

    add_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/add_edges/clustering_histogram/add_edges_mmds.csv"
        )
    )[relevant_cols]
    add_edges = add_edges.assign(perturb_type="Add Edges")

    gaussian_noise = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/gaussian_noise/clustering_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    gaussian_noise = gaussian_noise.assign(perturb_type="Gaussian Noise")

    remove_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/remove_edges/clustering_histogram/removedge_mmds.csv"
        )
    )[relevant_cols]
    remove_edges = remove_edges.assign(perturb_type="Remove Edges")

    rewire_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/rewire_edges/clustering_histogram/rewireedge_mmds.csv"
        )
    )[relevant_cols]
    rewire_edges = rewire_edges.assign(perturb_type="Rewire Edges")

    shear = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/shear/clustering_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    shear = shear.assign(perturb_type="Shear")

    taper = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/taper/clustering_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    taper = taper.assign(perturb_type="Taper")

    twist = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/twist/clustering_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    twist = twist.assign(perturb_type="Twist")

    all_data = pd.concat(
        [
            add_edges,
            gaussian_noise,
            remove_edges,
            rewire_edges,
            shear,
            taper,
            twist,
        ]
    )
    all_data = all_data.rename(
        columns={
            "perturb": "Perturbation",
            "sigma=0.01": "MMD",
            "perturb_type": "Perturbation Type",
        }
    )
    all_data = all_data.assign(descriptor="Clustering Histogram")
    return all_data


def load_degree():

    add_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/add_edges/degree_histogram/add_edges_mmds.csv"
        )
    )[relevant_cols]
    add_edges.sort_values(by=["perturb", "run"], inplace=True)
    add_edges = add_edges.assign(perturb_type="Add Edges")

    # gaussian_noise = normalize(
    #     pd.read_csv(
    #         here()
    #         / "data/systematic/human/fixed_length_kernels/eps_graph/8/gaussian_noise/degree_histogram/gaussian_noise_mmds.csv"
    #     )
    # )[relevant_cols]
    # gaussian_noise = gaussian_noise.assign(perturb_type="Gaussian Noise")

    remove_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/remove_edges/degree_histogram/remove_edges_mmds.csv"
        )
    )[relevant_cols]
    remove_edges = remove_edges.assign(perturb_type="Remove Edges")

    rewire_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/rewire_edges/degree_histogram/rewire_edges_mmds.csv"
        )
    )[relevant_cols]
    rewire_edges = rewire_edges.assign(perturb_type="Rewire Edges")

    # shear = normalize(
    #     pd.read_csv(
    #         here()
    #         / "data/systematic/human/fixed_length_kernels/eps_graph/8/shear/degree_histogram/shear_mmds.csv"
    #     )
    # )[relevant_cols]
    # shear = shear.assign(perturb_type="Shear")

    # taper = normalize(
    #     pd.read_csv(
    #         here()
    #         / "data/systematic/human/fixed_length_kernels/eps_graph/8/taper/degree_histogram/taper_mmds.csv"
    #     )
    # )[relevant_cols]
    # taper = taper.assign(perturb_type="Taper")

    # twist = normalize(
    #     pd.read_csv(
    #         here()
    #         / "data/systematic/human/fixed_length_kernels/eps_graph/8/twist/degree_histogram/twist_mmds.csv"
    #     )
    # )[relevant_cols]
    # twist = twist.assign(perturb_type="Twist")

    all_data = pd.concat(
        [
            add_edges,
            # gaussian_noise,
            remove_edges,
            rewire_edges,
            # shear,
            # taper,
            # twist,
        ]
    )
    all_data = all_data.rename(
        columns={
            "perturb": "Perturbation",
            "sigma=0.01": "MMD",
            "perturb_type": "Perturbation Type",
        }
    )
    all_data = all_data.assign(descriptor="Degree Histogram")
    return all_data


def load_laplacian():

    add_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/add_edges/laplacian_spectrum_histogram/add_edges_mmds.csv"
        )
    )[relevant_cols]
    add_edges.sort_values(by=["perturb", "run"], inplace=True)
    add_edges = add_edges.assign(perturb_type="Add Edges")

    gaussian_noise = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/gaussian_noise/laplacian_spectrum_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    gaussian_noise = gaussian_noise.assign(perturb_type="Gaussian Noise")

    remove_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/remove_edges/laplacian_spectrum_histogram/remove_edges_mmds.csv"
        )
    )[relevant_cols]
    remove_edges = remove_edges.assign(perturb_type="Remove Edges")

    rewire_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/rewire_edges/laplacian_spectrum_histogram/rewire_edges_mmds.csv"
        )
    )[relevant_cols]
    rewire_edges = rewire_edges.assign(perturb_type="Rewire Edges")

    shear = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/shear/laplacian_spectrum_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    shear = shear.assign(perturb_type="Shear")

    taper = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/taper/laplacian_spectrum_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    taper = taper.assign(perturb_type="Taper")

    twist = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/twist/laplacian_spectrum_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    twist = twist.assign(perturb_type="Twist")

    all_data = pd.concat(
        [
            add_edges,
            gaussian_noise,
            remove_edges,
            rewire_edges,
            shear,
            taper,
            twist,
        ]
    )

    all_data = all_data.rename(
        columns={
            "perturb": "Perturbation",
            "sigma=0.01": "MMD",
            "perturb_type": "Perturbation Type",
        }
    )
    all_data = all_data.assign(descriptor="Laplacian Spectrum Histogram")
    return all_data


def annotate(data, palette, **kws):
    for i, descriptor in enumerate(data.Descriptor.unique()):
        descriptor_df = data[data["Descriptor"] == descriptor]
        r_ps = list()
        r_ss = list()
        for run in descriptor_df.run.unique():
            run_df = descriptor_df[descriptor_df["run"] == run]
            r_p, p_p = sp.stats.pearsonr(
                run_df["Perturbation (%)"], run_df["Normalized MMD"]
            )
            r_ps.append(r_p)
            r_s, p_s = sp.stats.spearmanr(
                run_df["Perturbation (%)"], run_df["Normalized MMD"]
            )
            r_ss.append(r_s)

        avg_rp = np.mean(r_ps)
        avg_rs = np.mean(r_ss)
        ax = plt.gca()
        ax.text(
            0.8 - 0.23 * i,
            0.05,
            f"{descriptor.split(' ')[:-1][0]}" + "\n"
            r"$\rho_P=$"
            + f"{round(avg_rp, 2)}"
            + "\n"
            + r"$\rho_S=$"
            + f"{round(avg_rs, 2)}",
            # color=palette[i],
            transform=ax.transAxes,
            fontsize=8,
        )


def main():
    df_clustering = load_clustering()
    df_degree = load_degree()
    df_laplacian = load_laplacian()
    df = pd.concat([df_clustering, df_degree, df_laplacian])
    df.rename(
        columns={
            "descriptor": "Descriptor",
            "Perturbation": "Perturbation (%)",
            "MMD": "Normalized MMD",
        },
        inplace=True,
    )
    # Plot relplot of kind "line"
    setup_plotting_parameters(resolution=100)
    palette = sns.color_palette("mako_r", df["Descriptor"].nunique())

    df.reset_index(drop=True, inplace=True)
    g = sns.relplot(
        x="Perturbation (%)",
        y="Normalized MMD",
        hue="Descriptor",
        col="Perturbation Type",
        kind="line",
        data=df,
        height=3,
        aspect=0.75,
        col_wrap=3,
        palette=palette,
        ci=100,
        # facet_kws={"sharex": False},
    )
    g.map_dataframe(annotate, palette=palette)

    leg = g._legend
    leg.set_bbox_to_anchor([0.67, 0.2])

    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title(f"{df['Perturbation Type'].unique()[i]}")
    # g.fig.suptitle(
    #     r"MMD vs. Perturbation (%) For Various Graph Descriptors of the 8$\AA$-Graphs Under Different Perturbations Regimes."
    # )

    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(here() / "exploring/systematic_analysis/res_1_1.pdf")

    # sns.lineplot(data=add_edges, x="perturb", y="sigma=0.01")
    # sns.lineplot(data=remove_edges, x="perturb", y="sigma=0.01")
    # sns.lineplot(data=rewire_edges, x="perturb", y="sigma=0.01")
    # sns.lineplot(data=shear, x="perturb", y="sigma=0.01")
    # sns.lineplot(data=taper, x="perturb", y="sigma=0.01")
    # sns.lineplot(data=twist, x="perturb", y="sigma=0.01")
    # sns.lineplot(data=gaussian_noise, x="perturb", y="sigma=0.01")
    # plt.title("Clustering Histogram")


if __name__ == "__main__":
    main()
