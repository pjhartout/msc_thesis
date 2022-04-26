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

plt.rcParams["figure.figsize"] = (10.4, 6.8)
plt.rcParams["savefig.dpi"] = 800
mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(
    fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf"
)
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf_4")
def main(cfg: DictConfig):
    os.makedirs(here() / cfg.experiments.results / "images", exist_ok=True)

    df_plot = pd.DataFrame(columns=["mmd", "size", "organism"])
    for fname in os.listdir(here() / cfg.experiments.results):
        if "human" in fname:
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
    plt.legend(title="Organism")
    p.set_xlabel(r"Size of each set of proteins.")
    p.set_ylabel("Maximum Mean Discrepancy")
    plt.title(
        "Maximum Mean Discrepancy of random proteins samples from the Human Proteome and E. Coli Proteome"
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
