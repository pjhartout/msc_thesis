#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_multiple_experiments.py

Plot multiple experiments

"""

import os
from pathlib import Path

import hydra
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


@hydra.main(config_path=str(here()) + "/conf", config_name="conf_1")
def main(cfg: DictConfig):
    # Make image directory
    image_dir = here() / cfg.experiments.results / "images"
    if not image_dir.exists():
        image_dir.mkdir(parents=True)

    df_plot = pd.DataFrame(columns=["mmd", "std", "epsilon"])
    for parent, current, files in os.walk(here() / cfg.experiments.results):
        for file in files:
            if "epsilon" in file:
                exp_run = pd.read_csv(Path(parent) / file, index_col=0,)
                exp_run["epsilon"] = int(
                    ".".join(file.split("_")).split(".")[-2]
                )
                exp_run["run"] = int(parent.split("/")[-1])
                df_plot = pd.concat([df_plot, exp_run])

    print("Plotting")
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
        # err_style="bars",
        ci=100,
    )
    p = setup_annotations(
        p,
        title=r"MMD vs. Gaussian Noise Added to Different Sets of Proteins across varying values of $\varepsilon$.",
        x_label=r"Standard Deviation of Injected Noise ($\mathring{A}$)",
        y_label="Maximum Mean Discrepancy",
        legend_title=r"$\varepsilon$",
    )
    print("Saving")
    plt.tight_layout()
    plt.savefig(
        here() / cfg.experiments.results / "images" / Path("gaussian.png")
    )


if __name__ == "__main__":
    main()
