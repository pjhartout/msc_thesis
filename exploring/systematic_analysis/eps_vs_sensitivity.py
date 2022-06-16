#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""eps_vs_sensitivity.py

How does the eps threshold influence the sensitivity of MMD to perturbations?

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyparsing import col
from pyprojroot import here
from yaml import load

from proteinmetrics.utils.plots import setup_plotting_parameters

relevant_cols = ["perturb", "run", "sigma=0.01"]


def normalize(df):
    cols = [col for col in df.columns if "run" not in col]
    for col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def load_gaussian_clustering():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/gaussian_noise/clustering_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Clustering Histogram",
        perturb_type="Gaussian Noise",
        eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/gaussian_noise/clustering_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Clustering Histogram",
        perturb_type="Gaussian Noise",
        eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/gaussian_noise/clustering_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Clustering Histogram",
        perturb_type="Gaussian Noise",
        eps_value=32,
    )
    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_taper_clustering():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/taper/clustering_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Clustering Histogram", perturb_type="Taper", eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/taper/clustering_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Clustering Histogram", perturb_type="Taper", eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/taper/clustering_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Clustering Histogram", perturb_type="Taper", eps_value=32,
    )
    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_twist_clustering():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/twist/clustering_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Clustering Histogram", perturb_type="Twist", eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/twist/clustering_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Clustering Histogram", perturb_type="Twist", eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/twist/clustering_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Clustering Histogram", perturb_type="Twist", eps_value=32,
    )
    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_shear_clustering():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/shear/clustering_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Clustering Histogram", perturb_type="Shear", eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/shear/clustering_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Clustering Histogram", perturb_type="Shear", eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/shear/clustering_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Clustering Histogram", perturb_type="Shear", eps_value=32,
    )
    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_clustering():
    df_gaussian = load_gaussian_clustering()
    df_taper = load_taper_clustering()
    df_twist = load_twist_clustering()
    df_shear = load_shear_clustering()
    df = pd.concat(
        [df_gaussian, df_taper, df_twist, df_shear], ignore_index=True
    )
    return df.assign(descriptor="Clustering Histogram")


def load_gaussian_degree():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/gaussian_noise/degree_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Degree Histogram",
        perturb_type="Gaussian Noise",
        eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/gaussian_noise/degree_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Degree Histogram",
        perturb_type="Gaussian Noise",
        eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/gaussian_noise/degree_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Degree Histogram",
        perturb_type="Gaussian Noise",
        eps_value=32,
    )
    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_taper_degree():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/taper/degree_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Degree Histogram", perturb_type="Taper", eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/taper/degree_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Degree Histogram", perturb_type="Taper", eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/taper/degree_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Degree Histogram", perturb_type="Taper", eps_value=32,
    )
    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_twist_degree():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/twist/degree_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Degree Histogram", perturb_type="Twist", eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/twist/degree_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Degree Histogram", perturb_type="Twist", eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/twist/degree_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Degree Histogram", perturb_type="Twist", eps_value=32,
    )
    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_shear_degree():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/shear/degree_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Degree Histogram", perturb_type="Shear", eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/shear/degree_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Degree Histogram", perturb_type="Shear", eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/shear/degree_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Degree Histogram", perturb_type="Shear", eps_value=32,
    )
    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_degree():
    df_gaussian = load_gaussian_degree()
    df_taper = load_taper_degree()
    df_twist = load_twist_degree()
    df_shear = load_shear_degree()
    df = pd.concat(
        [df_gaussian, df_taper, df_twist, df_shear], ignore_index=True
    )
    return df.assign(descriptor="Degree Histogram")


#!/usr/bin/env python
# -*- coding: utf-8 -*-


relevant_cols = ["perturb", "run", "sigma=0.01"]


def normalize(df):
    cols = [col for col in df.columns if "run" not in col]
    for col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def load_gaussian_laplacian():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/gaussian_noise/laplacian_spectrum_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Gaussian Noise",
        eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/gaussian_noise/laplacian_spectrum_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Gaussian Noise",
        eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/gaussian_noise/laplacian_spectrum_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Gaussian Noise",
        eps_value=32,
    )
    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_taper_laplacian():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/taper/laplacian_spectrum_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Taper",
        eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/taper/laplacian_spectrum_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Taper",
        eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/taper/laplacian_spectrum_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Taper",
        eps_value=32,
    )
    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_twist_laplacian():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/twist/laplacian_spectrum_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Twist",
        eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/twist/laplacian_spectrum_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Twist",
        eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/twist/laplacian_spectrum_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Twist",
        eps_value=32,
    )
    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_shear_laplacian():
    clustering_eps_8 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/8/shear/laplacian_spectrum_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_8 = clustering_eps_8.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Shear",
        eps_value=8,
    )

    clustering_eps_16 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/16/shear/laplacian_spectrum_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_16 = clustering_eps_16.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Shear",
        eps_value=16,
    )

    clustering_eps_32 = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/eps_graph/32/shear/laplacian_spectrum_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    clustering_eps_32 = clustering_eps_32.assign(
        descriptor="Laplacian Spectrum Histogram",
        perturb_type="Shear",
        eps_value=32,
    )

    df = pd.concat(
        [clustering_eps_8, clustering_eps_16, clustering_eps_32,],
        ignore_index=True,
    )
    return df


def load_laplacian():
    df_gaussian = load_gaussian_laplacian()
    df_taper = load_taper_laplacian()
    df_twist = load_twist_laplacian()
    df_shear = load_shear_laplacian()
    df = pd.concat(
        [df_gaussian, df_taper, df_twist, df_shear], ignore_index=True
    )
    return df.assign(descriptor="Laplacian Spectrum Histogram")


def main():
    df_clustering = load_clustering()
    df_degree = load_degree()
    df_laplacian = load_laplacian()

    df = pd.concat([df_clustering, df_degree, df_laplacian], ignore_index=True)
    # df_addedges = load_addedges()
    # df_removeedges = load_removeedges()
    # df_rewireedges = load_rewireedges()

    df["combo"] = df["perturb_type"] + "_" + df["descriptor"]

    df = df.rename(
        columns={"perturb": "Perturbation", "sigma=0.01": "Normalized MMD"}
    )
    setup_plotting_parameters(resolution=100)
    palette = sns.color_palette("mako_r", df["eps_value"].nunique())

    df.reset_index(drop=True, inplace=True)
    g = sns.relplot(
        x="Perturbation",
        y="Normalized MMD",
        hue="eps_value",
        col="combo",
        kind="line",
        data=df,
        height=3,
        aspect=1,
        col_wrap=3,
        ci=100,
        palette=palette,
        col_order=[
            "Gaussian Noise_Clustering Histogram",
            "Gaussian Noise_Degree Histogram",
            "Gaussian Noise_Laplacian Spectrum Histogram",
            "Taper_Clustering Histogram",
            "Taper_Degree Histogram",
            "Taper_Laplacian Spectrum Histogram",
            "Shear_Clustering Histogram",
            "Shear_Degree Histogram",
            "Shear_Laplacian Spectrum Histogram",
            "Twist_Clustering Histogram",
            "Twist_Degree Histogram",
            "Twist_Laplacian Spectrum Histogram",
        ]
        # facet_kws={"legend_out": True}
        # facet_kws={"sharex": False},
    )
    g.fig.suptitle(
        r"Sensitivity of MMD to perturbations with varying thresholds $\varepsilon$.",
    )
    title = [
        "Gaussian Noise\n Clustering Histogram",
        "Gaussian Noise\n Degree Histogram",
        "Gaussian Noise\n  Laplacian Spectrum Histogram",
        "Taper\n Clustering Histogram",
        "Taper\n Degree Histogram",
        "Taper\n Laplacian Spectrum Histogram",
        "Shear\n Clustering Histogram",
        "Shear\n Degree Histogram",
        "Shear\n Laplacian Spectrum Histogram",
        "Twist\n Clustering Histogram",
        "Twist\n Degree Histogram",
        "Twist\n Laplacian Spectrum Histogram",
    ]
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title(f"{title[i]}")

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    # leg = g._legend
    # leg.set_bbox_to_anchor([1, 0.5])
    # plt.legend([], [], frameon=False)
    g._legend.set_title(r"$\varepsilon$-value" + "\n" + "(in $\AA$)")
    g.tight_layout(rect=[0, 0, 0.93, 1.0])
    plt.savefig(here() / "exploring/systematic_analysis/res_2.pdf")


if __name__ == "__main__":
    main()
