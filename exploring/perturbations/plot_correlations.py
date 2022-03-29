#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_correlations.py

Plots correlations as we vary the value of one of the parameters in the feature
extraction pipeline.

"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyprojroot import here

from proteinggnnmetrics.utils.functions import configure, flatten_lists

config = configure()

plt.rcParams["figure.figsize"] = (6.4, 4.8)
plt.rcParams["savefig.dpi"] = 1200
mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(
    fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf"
)
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False


def main():

    # Load the data
    results = pd.read_csv(
        here() / config["EXPERIMENT"]["CORR_EXPERIMENT_PATH"], index_col=0
    )
    spear_corr_name = "Spearman's Correlation Coefficient"
    pears_corr_name = "Pearson's Correlation Coefficient"
    epsilon_name = r"$\varepsilon$"
    results.columns = [epsilon_name, spear_corr_name, pears_corr_name]

    results = results.set_index(epsilon_name)
    results = results.reset_index().melt(
        id_vars=[epsilon_name], value_vars=[spear_corr_name, pears_corr_name]
    )
    palette = sns.color_palette("mako_r", len(results["variable"].unique()))
    p = sns.lineplot(
        data=results,
        x=epsilon_name,
        y="value",
        hue="variable",
        palette=palette,
        # style="choice",
    )
    plt.legend(title="Coorelation Coefficient")
    p.set_xlabel(epsilon_name)
    p.set_ylabel("Correlation Coefficient")
    # add title
    plt.title(f"Correlation coefficient vs {epsilon_name}")
    print("Saving")
    plt.tight_layout()
    plt.savefig(
        here()
        / config["EXPERIMENT"]["PLOT_PATH"]
        / Path("plot_correlations.png")
    )


if __name__ == "__main__":
    main()
