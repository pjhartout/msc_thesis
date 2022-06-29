#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""protein_specific_descriptors.py

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from pyprojroot import here
from yaml import load

from proteinmetrics.utils.plots import setup_plotting_parameters

relevant_cols = [
    "perturb",
    "run",
    "sigma=1e-05",
    # "sigma=0.0001",
    # "sigma=0.001",
    # "sigma=0.01",
    # "sigma=0.1",
    "sigma=1",
    # "sigma=10.0",
    # "sigma=100.0",
    # "sigma=1000.0",
    # "sigma=10000.0",
    # "sigma=100000.0",
    "linear_kernel",
]


def normalize(df):
    cols = [col for col in df.columns if "run" not in col]
    for col in cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def load_dihedral_angles():
    df_gaussian = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/pc_descriptor/1/gaussian_noise/dihedral_angles_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    df_gaussian = df_gaussian.assign(perturb_type="gaussian_noise")

    df_taper = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/pc_descriptor/1/taper/dihedral_angles_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    df_taper = df_taper.assign(perturb_type="taper")

    df_twist = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/pc_descriptor/1/twist/dihedral_angles_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    df_twist = df_twist.assign(perturb_type="twist")

    df_shear = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/pc_descriptor/1/shear/dihedral_angles_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    df_shear = df_shear.assign(perturb_type="shear")

    df = pd.concat(
        [df_gaussian, df_taper, df_twist, df_shear], ignore_index=True
    )
    return df


def load_distance_histogram():
    df_gaussian = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/pc_descriptor/1/gaussian_noise/distance_histogram/gaussian_noise_mmds.csv"
        )
    )[relevant_cols]
    df_gaussian = df_gaussian.assign(perturb_type="gaussian_noise")

    df_taper = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/pc_descriptor/1/taper/distance_histogram/taper_mmds.csv"
        )
    )[relevant_cols]
    df_taper = df_taper.assign(perturb_type="taper")

    df_twist = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/pc_descriptor/1/twist/distance_histogram/twist_mmds.csv"
        )
    )[relevant_cols]
    df_twist = df_twist.assign(perturb_type="twist")

    df_shear = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/pc_descriptor/1/shear/distance_histogram/shear_mmds.csv"
        )
    )[relevant_cols]
    df_shear = df_shear.assign(perturb_type="shear")

    df = pd.concat(
        [df_gaussian, df_taper, df_twist, df_shear], ignore_index=True
    )
    return df


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
        if descriptor == "Dihedral Angles Histogram":
            header = "Angles"
        else:
            header = "Distances"
        ax.text(
            0.8 - 0.23 * i,
            0.05,
            f"{header}" + "\n"
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
    df_dihedral = load_dihedral_angles()
    df_dihedral = df_dihedral.assign(descriptor="dihedral_angles",)
    df_distance = load_distance_histogram()
    df_distance = df_distance.assign(descriptor="distance_histogram",)

    df = pd.concat([df_distance, df_dihedral], ignore_index=True)
    df = df.melt(id_vars=["perturb_type", "run", "descriptor", "perturb"])
    df = df.rename(
        columns={
            "variable": "kernel",
            "value": "Normalized MMD",
            "perturb": "Perturbation (%)",
            "descriptor": "Descriptor",
        }
    )
    df["combo"] = df["perturb_type"] + "_" + df["kernel"]
    # Replace distance_histogram with Distance Histogram
    df["Descriptor"] = df["Descriptor"].str.replace(
        "distance_histogram", "Distance Histogram"
    )
    df["Descriptor"] = df["Descriptor"].str.replace(
        "dihedral_angles", "Dihedral Angles Histogram"
    )

    setup_plotting_parameters(resolution=100)
    palette = sns.color_palette("mako_r", df["Descriptor"].nunique())

    df.reset_index(drop=True, inplace=True)
    df["Perturbation (%)"] = df["Perturbation (%)"] * 100
    g = sns.relplot(
        x="Perturbation (%)",
        y="Normalized MMD",
        hue="Descriptor",
        col="combo",
        kind="line",
        data=df,
        height=3,
        aspect=0.75,
        col_wrap=3,
        ci=100,
        palette=palette,
        col_order=df.sort_values(by=["perturb_type", "kernel"]).combo.unique(),
    )
    g.map_dataframe(annotate, palette=palette)
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=3,
        title=None,
        frameon=False,
    )
    g._legend.set_title(r"Protein Descriptor")
    g.tight_layout(rect=[0, 0.045, 1, 0.99])

    titles = [
        "Gaussian Noise" + "\n" + "Linear Kernel",
        "Gaussian Noise" + "\n" + r"RBF Kernel $\sigma=1$",
        "Gaussian Noise" + "\n" + r"RBF Kernel $\sigma=10^{-5}$",
        "Shear" + "\n" + "Linear Kernel",
        "Shear" + "\n" + r"RBF Kernel $\sigma=1$",
        "Shear" + "\n" + r"RBF Kernel $\sigma=10^{-5}$",
        "Taper" + "\n" + "Linear Kernel",
        "Taper" + "\n" + r"RBF Kernel $\sigma=1$",
        "Taper" + "\n" + r"RBF Kernel $\sigma=10^{-5}$",
        "Twist" + "\n" + "Linear Kernel",
        "Twist" + "\n" + r"RBF Kernel $\sigma=1$",
        "Twist" + "\n" + r"RBF Kernel $\sigma=10^{-5}$",
    ]
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title(titles[i])

    plt.subplots_adjust(hspace=0.3)
    plt.savefig(here() / "exploring/systematic_analysis/res_4.pdf")


if __name__ == "__main__":
    main()
