#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_experiment.py

This file takes in the csv results of the experiment and makes a linechart out of it.

"""

from pathlib import Path

import hydra
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here

from proteinggnnmetrics.paths import CACHE_DIR
from proteinggnnmetrics.utils.functions import flatten_lists

plt.rcParams["figure.figsize"] = (6.4, 4.8)
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
    # Load the data
    results = pd.read_csv(
        here()
        / cfg.paths.experiment
        / "results_epsilon_4_20220327-214236.csv",
        index_col=0,
    )
    mmd_name = "Maximum Mean Discrepancy"
    gnoise_name = "Injected Gaussian Noise to Coordinates"
    results.columns = [mmd_name, gnoise_name]

    results = results.set_index(gnoise_name)
    p = sns.lineplot(
        data=results,
        x=gnoise_name,
        y=mmd_name,
        # hue="coherence",
        # style="choice",
    )
    p.set_xlabel(r"Standard Deviation of Injected Noise")
    p.set_ylabel("Maximum Mean Discrepancy")
    # add title
    plt.title(
        f"{mmd_name} computed from Weisfeiler-Lehman kernel \n vs {gnoise_name}."
    )
    print("Saving")
    plt.tight_layout()
    plt.savefig(
        here() / cfg.paths.experiment / "plots" / "plot_experiment.png"
    )


if __name__ == "__main__":
    main()
