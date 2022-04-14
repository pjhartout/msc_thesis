#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_experiment.py

Plot the results of the experiment.

"""

import os

import hydra
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from pyprojroot import here

plt.rcParams["figure.figsize"] = (8.4, 4.8)
plt.rcParams["savefig.dpi"] = 1200
mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(
    fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf"
)
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf")
def main(cfg: DictConfig):
    # Load the data
    df = pd.read_csv(
        here() / cfg.experiments.results / "mmd_single_run_twist.csv",
        index_col=0,
        header=0,
    )

    # Plot the data
    p = sns.lineplot(data=df, x="twist", y="mmd",)
    p.set_xlabel("Twist")
    p.set_ylabel("MMD")
    plt.title("MMD as twist is added to a different set of proteins.")
    plt.tight_layout()
    os.makedirs(here() / cfg.experiments.results / "images", exist_ok=True)
    plt.savefig(here() / cfg.experiments.results / "images/results.png")


if __name__ == "__main__":
    main()
