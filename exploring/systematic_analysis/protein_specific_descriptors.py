#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""protein_specific_descriptors.py

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyprojroot import here
from yaml import load

from proteinmetrics.utils.plots import setup_plotting_parameters

relevant_cols = ["perturb", "run", "sigma=0.01"]


def normalize(df):
    cols = [col for col in df.columns if "run" not in col]
    for col in df.columns:
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


def main():
    df_dihedral = load_dihedral_angles()
    df_dihedral = df_dihedral.assign(descriptor="dihedral_angles",)
    df_distance = load_dihedral_angles()
    df_distance = df_distance.assign(descriptor="distance_histogram",)

    df = pd.concat([df_dihedral], ignore_index=True)
    # df_addedges = load_addedges()
    # df_removeedges = load_removeedges()
    # df_rewireedges = load_rewireedges()

    setup_plotting_parameters(resolution=100)
    palette = sns.color_palette("mako_r", df["descriptor"].nunique())

    df.reset_index(drop=True, inplace=True)
    g = sns.relplot(
        x="perturb",
        y="sigma=0.01",
        hue="descriptor",
        col="perturb_type",
        kind="line",
        data=df,
        height=3,
        aspect=0.9,
        col_wrap=4,
        ci=100,
        palette=palette,
    )
    g.fig.suptitle(r"Protein Descriptors",)
    g.tight_layout(rect=[0, 0, 0.95, 1.0])
    plt.savefig(here() / "exploring/systematic_analysis/res_3.svg")


if __name__ == "__main__":
    main()
