#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""comparing_graph_construction_techniques.py

"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import seaborn as sns
from pyprojroot import here

from proteinmetrics.utils.plots import setup_plotting_parameters

eps_range = [8, 16, 32]
k_range = [6, 8, 16]
perturbations = [
    "add_edges",
    "gaussian_noise",
    "remove_edges",
    "rewire_edges",
    "shear",
    "taper",
    "twist",
]
descriptors = [
    "clustering_histogram",
    "degree_histogram",
    "laplacian_spectrum_histogram",
]
graph_types = ["eps", "knn"]


def normalize(df):
    cols = [col for col in df.columns if "run" not in col]
    for col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def load_data(graph_type, construct_param, perturb, descriptor):
    if Path(
        here()
        / "data"
        / "systematic"
        / "human"
        / "fixed_length_kernels"
        / f"{graph_type}_graph"
        / f"{construct_param}"
        / f"{perturb}"
        / f"{descriptor}"
    ).exists():
        return normalize(
            pd.read_csv(
                here()
                / "data"
                / "systematic"
                / "human"
                / "fixed_length_kernels"
                / f"{graph_type}_graph"
                / f"{construct_param}"
                / f"{perturb}"
                / f"{descriptor}"
                / f"{perturb}_mmds.csv"
            )
        ).assign(
            perturb_type=perturb,
            descriptor=descriptor,
            graph_type=graph_type,
            construct_param=construct_param,
        )
    else:
        print("No data for", graph_type, construct_param, perturb, descriptor)
        return pd.DataFrame()


def load_graph_construction(graph_type, construct_param):
    graph_construction_df = pd.DataFrame()
    for perturb in perturbations:
        for descriptor in descriptors:
            df = load_data(graph_type, construct_param, perturb, descriptor)
            graph_construction_df = pd.concat([graph_construction_df, df])
    return graph_construction_df


def load_eps():
    eps_df = pd.DataFrame()
    for eps in eps_range:
        df = load_graph_construction(graph_type="eps", construct_param=eps)
        eps_df = pd.concat([eps_df, df])
    return eps_df


def load_k():
    k_df = pd.DataFrame()
    for k in k_range:
        df = load_graph_construction(graph_type="knn", construct_param=k)
        k_df = pd.concat([k_df, df])
    return k_df


def compute_correlations(df, construct_param_range, graph_type):
    df_corr = pd.DataFrame()
    for param in construct_param_range:
        for descriptor in descriptors:
            for perturb in perturbations:
                for run in range(10):
                    if not df.loc[
                        (df.construct_param == param)
                        & (df.descriptor == descriptor)
                        & (df.perturb_type == perturb)
                        & (df.run == run),
                    ]["Normalized MMD"].empty:
                        ps = sp.stats.spearmanr(
                            df.loc[
                                (df.construct_param == param)
                                & (df.descriptor == descriptor)
                                & (df.perturb_type == perturb)
                                & (df.run == run),
                                "Normalized MMD",
                            ].values,
                            df.loc[
                                (df.construct_param == param)
                                & (df.descriptor == descriptor)
                                & (df.perturb_type == perturb)
                                & (df.run == run),
                                "perturb",
                            ].values,
                        )[0]
                        pp = sp.stats.pearsonr(
                            df.loc[
                                (df.construct_param == param)
                                & (df.descriptor == descriptor)
                                & (df.perturb_type == perturb)
                                & (df.run == run),
                                "Normalized MMD",
                            ].values,
                            df.loc[
                                (df.construct_param == param)
                                & (df.descriptor == descriptor)
                                & (df.perturb_type == perturb)
                                & (df.run == run),
                                "perturb",
                            ].values,
                        )[0]
                        df_corr = pd.concat(
                            [
                                df_corr,
                                pd.DataFrame(
                                    columns=[
                                        "graph_type",
                                        "param",
                                        "descriptor",
                                        "perturb_type",
                                        "run",
                                        "spearman",
                                        "pearson",
                                    ],
                                    data=[
                                        [
                                            graph_type,
                                            param,
                                            descriptor,
                                            perturb,
                                            run,
                                            ps,
                                            pp,
                                        ]
                                    ],
                                ),
                            ]
                        )
    return df_corr


def main():
    setup_plotting_parameters(size=(5.8, 4.8))
    # Load data
    eps_df = load_eps()
    k_df = load_k()
    df = pd.concat([eps_df, k_df])

    # Get relevant kernel
    df = df.rename(columns={"sigma=0.01": "Normalized MMD"})
    df = df[
        [
            col
            for col in df.columns
            if "sigma" not in col and "linear_kernel" not in col
        ]
    ]

    eps_df = df.loc[df.graph_type == "eps"]
    eps_corr = compute_correlations(eps_df, eps_range, "eps")

    k_df = df.loc[df.graph_type == "knn"]
    k_corr = compute_correlations(k_df, k_range, "knn")

    corr = pd.concat([eps_corr, k_corr])
    corr = corr.melt(
        id_vars=["graph_type", "param", "descriptor", "perturb_type", "run"]
    )

    g = sns.violinplot(
        x="graph_type",
        y="value",
        hue="variable",
        data=corr,
        palette=sns.color_palette("mako_r", n_colors=2),
        split=True,
        size=3,
        bw=0.1,
        # dodge_order=perturbations,
        # size=8,
        # linewidth=0.5,
        # edgecolor="black",
        # alpha=0.5,
    )
    g.set_xlabel("Graph Type")
    g.set_ylabel("Correlation Coefficient")
    g.set_xticklabels([r"$\varepsilon$-graphs", r"$k$-NN graphs"])
    g.get_legend().set_title("Correlation Type")
    sns.move_legend(g, "lower left", bbox_to_anchor=(0.28, 0))

    new_labels = [
        r"$\rho_S$ $\mu_{k}=0.879$, $\mu_{\varepsilon}=0.959$",
        r"$\rho_P$ $\mu_{k}=0.877$, $\mu_{\varepsilon}=0.914$",
    ]
    for t, l in zip(g.get_legend().texts, new_labels):
        t.set_text(l)
    # statistical annotation
    x1, x2 = (
        0.2,
        1.2,
    )
    y, h, col = 1 + 0.03, 0.1, "k"
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text(
        (x1 + x2) * 0.5, y + h, "***", ha="center", va="bottom", color=col,
    )

    x1, x2 = (
        -0.2,
        0.8,
    )
    y, h, col = 1 + 0.1, 0.2, "k"
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text(
        (x1 + x2) * 0.5, y + h, "*", ha="center", va="bottom", color=col,
    )

    # p_spearman: 0.0008349766424206454
    # p_pearson: 0.017589242617265154

    # # statistical annotation
    # x1, x2 = (
    #     1,
    #     3,
    # )  # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    # y, h, col = 1 + 2, 2, "k"
    # plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # plt.text(
    #     (x1 + x2) * 0.5,
    #     y + h,
    #     "p=0.01759",
    #     ha="center",
    #     va="bottom",
    #     color=col,
    # )
    # g.set_xlabel("Colors")
    # g.set_ylabel("Values")
    # g.set_yticklabels(["Red", "Green", "Blue"])

    plt.savefig(
        here()
        / "exploring"
        / "systematic_analysis"
        / "compare_corrs_graph_construction.pdf"
    )

    U1, p = sp.stats.mannwhitneyu(
        corr.loc[
            (corr.graph_type == "eps") & (corr.variable == "spearman")
        ].value.values,
        corr.loc[
            (corr.graph_type == "knn") & (corr.variable == "spearman")
        ].value.values,
        method="exact",
    )
    print(f"p_spearman: {p}")

    U1, p = sp.stats.mannwhitneyu(
        corr.loc[
            (corr.graph_type == "eps") & (corr.variable == "pearson")
        ].value.values,
        corr.loc[
            (corr.graph_type == "knn") & (corr.variable == "pearson")
        ].value.values,
        method="exact",
    )
    print(f"p_pearson: {p}")


if __name__ == "__main__":
    main()
    print("\nDone.")
