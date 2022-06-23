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

    mutation = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/weisfeiler_lehman/eps_graph/8/mutation/mutation_mmds.csv"
        )
    )
    mutation = twist.assign(perturb_type="Mutation")

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
    df = df.melt(id_vars=["run", "perturb", "perturb_type"])
    # Rename columns
    df = df.rename(
        columns={
            "variable": "Kernel Settings",
            "value": "Normalized MMDs",
            "perturb": "Perturbation (%)",
        }
    )
    # Filter n_iter=1, n_iter=5, n_iter=10
    df = df[df["Kernel Settings"].isin(["n_iter=1", "n_iter=5", "n_iter=10"])]
    # rename n_iter=1, n_iter=5, n_iter=10 to Iterations: 1, Iterations: 5, Iterations: 10
    df["Kernel Settings"] = df["Kernel Settings"].str.replace(
        "n_iter=", "Iterations: "
    )
    print("Loaded data")
    setup_plotting_parameters(resolution=100)
    palette = sns.color_palette("mako_r", df["Kernel Settings"].nunique())

    g = sns.relplot(
        x="Perturbation (%)",
        y="Normalized MMDs",
        col="perturb_type",
        hue="Kernel Settings",
        kind="line",
        data=df,
        height=3,
        aspect=1,
        col_wrap=3,
        ci=100,
        palette=palette,
        col_order=[
            "Add Edges",
            "Remove Edges",
            "Rewire Edges",
            "Gaussian Noise",
            "Mutation",
            "Shear",
            "Taper",
            "Twist",
        ]
        # facet_kws={"sharex": False},
    )

    title = [
        "Add Edges",
        "Remove Edges",
        "Rewire Edges",
        "Gaussian Noise",
        "Mutation",
        "Shear",
        "Taper",
        "Twist",
    ]
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title(f"{title[i]}")

    leg = g._legend
    leg.set_bbox_to_anchor([0.9, 0.2])
    plt.tight_layout()
    plt.savefig(here() / "exploring/systematic_analysis/res_3.pdf")


if __name__ == "__main__":
    main()
