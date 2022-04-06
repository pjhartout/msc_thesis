#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_multiple_runs.py

Plot curves for multiple runs

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

plt.rcParams["figure.figsize"] = (10.4, 6.8)
plt.rcParams["savefig.dpi"] = 1200
mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(
    fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf"
)
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False


@hydra.main(config_path=str(here()) + "/conf", config_name="config.yaml")
def main(cfg: DictConfig):

    df_plot = pd.DataFrame(columns=["mmd", "std", "epsilon", "run"])
    for run in range(cfg.eps_var.n_runs):
        df_run = pd.DataFrame(columns=["mmd", "std", "epsilon"])
        for fname in os.listdir(here() / cfg.eps_var.paths.results / str(run)):
            if "epsilon" in fname:
                df = pd.read_csv(
                    here() / cfg.eps_var.paths.results / str(run) / fname,
                    index_col=0,
                    header=0,
                )
                df["epsilon"] = int(".".join(fname.split("_")).split(".")[1])
                df_run = pd.concat([df_run, df], ignore_index=True)
        df_run["run"] = run
        df_run.reset_index(drop=True, inplace=True)
        df_plot.reset_index(drop=True, inplace=True)
        df_plot = pd.concat([df_plot, df_run])
    print("Plotting")
    df_plot.reset_index(drop=True, inplace=True)
    df_plot = df_plot.melt(id_vars=["epsilon", "mmd", "std"]).drop(
        columns=["variable", "value"]
    )
    df_plot = df_plot.sort_values(by=["epsilon", "std"])
    palette = sns.color_palette("mako_r", len(df_plot["epsilon"].unique()))
    p = sns.lineplot(
        data=df_plot,
        x="std",
        y="mmd",
        hue="epsilon",
        palette=palette,
        err_style="bars",
        ci=100,
    )
    plt.legend(title=r"$\varepsilon$")
    p.set_xlabel(r"Standard Deviation of Injected Noise")
    p.set_ylabel("Maximum Mean Discrepancy")
    plt.title(
        "Multiple experiments of MMD vs gausian noise injection \n with varying levels of epsilon."
    )
    print("Saving")
    plt.tight_layout()
    plt.savefig(
        here() / cfg.eps_var.paths.images / Path("plot_multiple_runs.png")
    )


if __name__ == "__main__":
    main()
