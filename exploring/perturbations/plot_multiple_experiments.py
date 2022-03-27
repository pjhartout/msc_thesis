#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_multiple_experiments.py

Plot multiple experiments

"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyprojroot import here

from proteinggnnmetrics.utils.functions import configure, flatten_lists

config = configure()

plt.rcParams["figure.figsize"] = (10.4, 6.8)


def main():
    df_plot = pd.DataFrame(columns=["mmd", "std", "epsilon"])
    for fname in os.listdir(here() / config["EXPERIMENT"]["EXPERIMENT_PATH"]):
        if "results" in fname:
            exp_run = pd.read_csv(
                here() / config["EXPERIMENT"]["EXPERIMENT_PATH"] / Path(fname),
                index_col=0,
            )
            exp_run["epsilon"] = int(".".join(fname.split("_")).split(".")[2])
            df_plot = pd.concat([df_plot, exp_run])
    print("Plotting")
    palette = sns.color_palette("mako_r", len(df_plot["epsilon"].unique()))
    sns.lineplot(
        data=df_plot.reset_index(drop=True),
        x="std",
        y="mmd",
        hue="epsilon",
        palette=palette,
    )
    plt.title(
        "Multiple experiments of MMD vs gausian noise injection \n with varying levels of epsilon."
    )
    print("Saving")
    plt.tight_layout()
    plt.savefig(
        config["EXPERIMENT"]["PLOT_PATH"]
        / Path("plot_multiple_experiments.png")
    )


if __name__ == "__main__":
    main()
