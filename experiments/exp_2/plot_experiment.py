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

from proteinggnnmetrics.utils.plots import setup_plotting_parameters

setup_plotting_parameters()


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf_2")
def main(cfg: DictConfig):
    # Load the data
    df_plot = pd.DataFrame(columns=["mmd_tda", "mmd_wl", "twist"])
    for parent, current, files in os.walk(here() / cfg.experiments.results):
        for file in files:
            if "mmd" in file:
                exp_run = pd.read_csv(Path(parent) / file, index_col=0,)
                exp_run["run"] = int(".".join(file.split("_")).split(".")[-2])
                df_plot = pd.concat([df_plot, exp_run])

    # Plot the data
    df_plot = df_plot.reset_index(drop=True)
    # Normalize mmd values
    df_plot = df_plot.assign(
        mmd_tda=(df_plot["mmd_tda"] - df_plot["mmd_tda"].min())
        / (df_plot["mmd_tda"].max() - df_plot["mmd_tda"].min()),
        mmd_wl=(df_plot["mmd_wl"] - df_plot["mmd_wl"].min())
        / (df_plot["mmd_wl"].max() - df_plot["mmd_wl"].min()),
    )
    df_plot = df_plot.rename(
        columns={
            "mmd_wl": r"MMD Weisfeiler-Lehman on 8$\mathring{A}$-graph",
            "mmd_tda": "MMD Persistence Fisher on persistence diagrams",
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
    plt.legend(title=r"Kernel")
    p.set_xlabel(r"Twist (rad/$\mathring{A}$)")
    p.set_ylabel("MMD (normalized)")
    plt.title("MMD as twist is added to a different set of proteins.")
    plt.tight_layout()
    os.makedirs(here() / cfg.experiments.results / "images", exist_ok=True)
    plt.savefig(here() / cfg.experiments.results / "images/results.png")


if __name__ == "__main__":
    main()
