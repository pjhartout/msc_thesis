#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_experiment_4.py

Plot results of experiment 4.

"""

import os
from multiprocessing.sharedctypes import Value
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


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf_4")
def main(cfg: DictConfig):
    os.makedirs(here() / cfg.experiments.results / "images", exist_ok=True)

    df_plot = pd.DataFrame(columns=["mmd", "size", "organism"])
    for fname in os.listdir(here() / cfg.experiments.results):
        if "ecoli" not in fname and "mmd" in fname:
            df = pd.read_csv(
                here() / cfg.experiments.results / fname,
                names=["mmd", "size"],
                header=0,
            )
            df["organism"] = "human"
            df_plot = pd.concat([df_plot, df])
        elif "ecoli" in fname:
            df = pd.read_csv(
                here() / cfg.experiments.results / fname,
                names=["mmd", "size"],
                header=0,
            )
            df["organism"] = "ecoli"
            df_plot = pd.concat([df_plot, df])
        else:
            continue

    # Fix typing
    df_plot = df_plot.astype(
        {"mmd": "float", "size": "int", "organism": "string"}
    )
    df_plot = df_plot.replace(
        {"organism": {"human": "Human", "ecoli": "E. Coli"}}
    )
    print("Plotting")
    p = sns.violinplot(
        x="size",
        y="mmd",
        hue="organism",
        data=df_plot,
        palette="mako_r",
        split=True,
    )
    p = setup_annotations(
        p,
        title="MMD of random samples from Human and E. Coli Proteome.",
        x_label="Size of each set of proteins.",
        y_label="Maximum Mean Discrepancy",
        legend_title="Organism",
    )

    print("Saving")
    plt.tight_layout()
    plt.savefig(
        here()
        / cfg.experiments.results
        / "images"
        / Path("mmd_variance_baseline.png")
    )


if __name__ == "__main__":
    main()
