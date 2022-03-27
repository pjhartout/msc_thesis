#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_correlations.py

Plots correlations as we vary the value of one of the parameters in the feature
extraction pipeline.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyprojroot import here

from proteinggnnmetrics.utils.functions import configure, flatten_lists

config = configure()

plt.rcParams["figure.figsize"] = (10.4, 6.8)


def main():

    # Load the data
    results = pd.read_csv(
        here() / config["EXPERIMENT"]["CORR_EXPERIMENT_PATH"], index_col=0
    )
    spear_corr_name = "Spearman's Correlation Coefficient"
    pears_corr_name = "Pearson's Correlation Coefficient"
    epsilon_name = "Epsilon value used to extract the graphs"
    results.columns = [epsilon_name, spear_corr_name, pears_corr_name]

    results = results.set_index(epsilon_name)
    results = results.reset_index().melt(
        id_vars=[epsilon_name], value_vars=[spear_corr_name, pears_corr_name]
    )
    palette = sns.color_palette("mako_r", len(results["variable"].unique()))
    sns.lineplot(
        data=results,
        x=epsilon_name,
        y="value",
        hue="variable",
        palette=palette,
        # style="choice",
    )
    # add title
    plt.title(f"{spear_corr_name} vs {epsilon_name}")
    print("Saving")
    plt.tight_layout()
    plt.savefig(
        config["EXPERIMENT"]["PLOT_PATH"] / Path("plot_correlations.png")
    )


if __name__ == "__main__":
    main()
