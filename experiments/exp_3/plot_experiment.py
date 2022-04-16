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

plt.rcParams["figure.figsize"] = (10.4, 6.8)
plt.rcParams["savefig.dpi"] = 1200
mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(
    fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf"
)
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False


@hydra.main(config_path=str(here()) + "/conf", config_name="conf")
def main(cfg: DictConfig):
    df_plot = pd.DataFrame(columns=["mmd", "twist"])
    for fname in os.listdir(here() / cfg.experiments.results):
        if "single_run" in fname:
            exp_run = pd.read_csv(
                here() / cfg.experiments.results / fname, index_col=0,
            )
            df_plot = pd.concat([df_plot, exp_run])
    print("Plotting")

    # palette = sns.color_palette("mako_r", len(df_plot["epsilon"].unique()))
    p = sns.lineplot(
        data=df_plot.reset_index(drop=True),
        x="twist",
        y="mmd",
        # palette=palette,
    )
    # plt.legend(title=r"$\varepsilon$")
    p.set_xlabel("Mutation probability")
    p.set_ylabel("Maximum Mean Discrepancy")
    plt.title(
        "Multiple experiments of MMD vs mutation as captured through ESM embeddings."
    )
    print("Saving")
    plt.tight_layout()
    plt.savefig(
        here()
        / cfg.experiments.results
        / "images"
        / Path("plot_multiple_experiments.png")
    )


if __name__ == "__main__":
    main()
