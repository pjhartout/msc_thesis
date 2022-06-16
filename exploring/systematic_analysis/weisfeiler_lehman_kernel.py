# -*- coding: utf-8 -*-
"""weisfeiler_lehman_kernel.py

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyprojroot import here
from torch import normal

from proteinmetrics.utils.plots import setup_plotting_parameters


def normalize(df):
    cols = [col for col in df.columns if "run" not in col]
    for col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def load_data():
    add_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/weisfeiler_lehman/eps_graph/8/add_edges/add_edges_mmds.csv"
        )
    )
    add_edges = add_edges.assign(perturb_type="Add Edges")

    gaussian_noise = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/weisfeiler_lehman/eps_graph/8/gaussian_noise/gaussian_noise_mmds.csv"
        )
    )
    gaussian_noise = gaussian_noise.assign(perturb_type="Gaussian Noise")

    mutation = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/weisfeiler_lehman/eps_graph/8/mutation/mutation_mmds.csv"
        )
    )
    mutation = mutation.assign(perturb_type="Mutation")

    remove_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/weisfeiler_lehman/eps_graph/8/remove_edges/remove_edges_mmds.csv"
        )
    )
    remove_edges = remove_edges.assign(perturb_type="Remove Edges")

    rewire_edges = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/weisfeiler_lehman/eps_graph/8/rewire_edges/rewire_edges_mmds.csv"
        )
    )
    rewire_edges = rewire_edges.assign(perturb_type="Rewire Edges")

    shear = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/weisfeiler_lehman/eps_graph/8/shear/shear_mmds.csv"
        )
    )
    shear = shear.assign(perturb_type="Shear")

    taper = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/weisfeiler_lehman/eps_graph/8/taper/taper_mmds.csv"
        )
    )
    taper = taper.assign(perturb_type="Taper")

    twist = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/weisfeiler_lehman/eps_graph/8/twist/twist_mmds.csv"
        )
    )
    twist = twist.assign(perturb_type="Twist")

    return pd.concat(
        [
            add_edges,
            gaussian_noise,
            mutation,
            remove_edges,
            rewire_edges,
            shear,
            taper,
            twist,
        ]
    ).reset_index(drop=True)


def main():
    df = load_data()
    print("Loaded data")
    setup_plotting_parameters(resolution=100)
    g = sns.relplot(
        x="perturb",
        y="n_iter=5",
        col="perturb_type",
        kind="line",
        data=df,
        height=3,
        aspect=1,
        col_wrap=4,
        ci=100,
        # facet_kws={"sharex": False},
    )
    plt.tight_layout()
    plt.savefig(here() / "exploring/systematic_analysis/res_4.svg")


if __name__ == "__main__":
    main()
