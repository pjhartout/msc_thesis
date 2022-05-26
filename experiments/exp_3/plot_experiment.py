#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_multiple_experiments.py

Plot multiple experiments of exp_3

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

from proteinmetrics.utils.plots import (
    setup_annotations,
    setup_plotting_parameters,
)

setup_plotting_parameters()


@hydra.main(config_path=str(here()) + "/conf", config_name="conf_3")
def main(cfg: DictConfig):
    df_plot = pd.DataFrame()

    for fname in os.listdir(here() / cfg.experiments.results):
        if "single_run" in fname and "other_kernels" not in fname:
            exp_run = pd.read_csv(
                here() / cfg.experiments.results / fname, index_col=0,
            )
            exp_run["run"] = int(".".join(fname.split("_")).split(".")[-2])
            df_plot = pd.concat([df_plot, exp_run])

    print("Plotting")
    df_plot = df_plot.assign(
        mmd_esm=(df_plot["mmd_esm"] - df_plot["mmd_esm"].min())
        / (df_plot["mmd_esm"].max() - df_plot["mmd_esm"].min()),
        mmd_wl=(df_plot["mmd_wl"] - df_plot["mmd_wl"].min())
        / (df_plot["mmd_wl"].max() - df_plot["mmd_wl"].min()),
    )

    df_plot = df_plot.rename(
        columns={
            "mmd_esm": "MMD with Linear Kernel from ESM embeddings",
            "mmd_wl": "MMD from Weisfeiler-Lehman kernel",
        }
    )
    df_plot = df_plot.melt(id_vars=["p_mutate", "run"])
    palette = sns.color_palette("mako_r", len(df_plot["variable"].unique()))
    p = sns.lineplot(
        data=df_plot.reset_index(drop=True),
        x="p_mutate",
        y="value",
        hue="variable",
        palette=palette,
        ci=100,
    )
    p = setup_annotations(
        p,
        title="MMD vs. Mutations Added to Different Sets of Proteins.",
        x_label="Mutation Probability",
        y_label="Maximum Mean Discrepancy (Normalized)",
        legend_title="Kernel",
    )

    print("Saving")
    os.makedirs(here() / cfg.experiments.results / "images", exist_ok=True)
    plt.savefig(
        here() / cfg.experiments.results / "images" / Path("mutation.png")
    )


if __name__ == "__main__":
    main()
