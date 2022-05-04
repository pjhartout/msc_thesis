#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_experiment.py

Plot the results of the experiment.

"""

import os
from pathlib import Path

import hydra
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from pyprojroot import here

from proteinggnnmetrics.utils.plots import (
    setup_annotations,
    setup_plotting_parameters,
)

setup_plotting_parameters()


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf_2")
def main(cfg: DictConfig):
    # Load the data
    df_plot = pd.DataFrame(columns=["mmd_tda", "mmd_wl", "twist"])
    for parent, current, files in os.walk(here() / cfg.experiments.results):
        for file in files:
            if "mmd" in file and "ramachandran" not in file:
                exp_run = pd.read_csv(Path(parent) / file, index_col=0,)
                exp_run["run"] = int(".".join(file.split("_")).split(".")[-2])
                df_plot = pd.concat([df_plot, exp_run])
            elif "ramachandran" in file:
                exp_run = pd.read_csv(Path(parent) / file, index_col=0,)
                exp_run["run"] = int(".".join(file.split("_")).split(".")[-3])
                df_plot = pd.concat([df_plot, exp_run])

    # Collapse dataframe
    df_plot = pd.merge(
        df_plot[["mmd_tda", "mmd_wl", "twist", "run"]].dropna(),
        df_plot[["mmd_rama", "twist", "run"]].dropna(),
        left_on=["run", "twist"],
        right_on=["run", "twist"],
        how="inner",
    )
    # Plot the data
    df_plot = df_plot.reset_index(drop=True)
    # Normalize mmd values
    df_plot = df_plot.assign(
        mmd_tda=(df_plot["mmd_tda"] - df_plot["mmd_tda"].min())
        / (df_plot["mmd_tda"].max() - df_plot["mmd_tda"].min()),
        mmd_wl=(df_plot["mmd_wl"] - df_plot["mmd_wl"].min())
        / (df_plot["mmd_wl"].max() - df_plot["mmd_wl"].min()),
        mmd_rama=(df_plot["mmd_rama"] - df_plot["mmd_rama"].min())
        / (df_plot["mmd_rama"].max() - df_plot["mmd_rama"].min()),
    )
    df_plot = df_plot.rename(
        columns={
            "mmd_wl": r"MMD Weisfeiler-Lehman Kernel on 8$\mathring{A}$-graph",
            "mmd_tda": "MMD Persistence Fisher Kernel on Persistence Diagrams",
            "mmd_rama": "MMD Linear Kernel from Ramachandran Descriptor",
        }
    )
    df_plot = df_plot.melt(id_vars=["twist", "run"])
    palette = sns.color_palette("mako_r", len(df_plot["variable"].unique()))
    p = sns.lineplot(
        data=df_plot.reset_index(drop=True),
        x="twist",
        y="value",
        hue="variable",
        palette=palette,
        ci=100,
    )
    p = setup_annotations(
        p,
        title="MMD vs. Twist Added to Different Sets of Proteins.",
        x_label=r"Twist (rad/$\mathring{A}$)",
        y_label="Maximum Mean Discrepancy (Normalized)",
        legend_title="Kernel",
    )

    plt.tight_layout()
    os.makedirs(here() / cfg.experiments.results / "images", exist_ok=True)
    plt.savefig(here() / cfg.experiments.results / "images/twist.png")


if __name__ == "__main__":
    main()
