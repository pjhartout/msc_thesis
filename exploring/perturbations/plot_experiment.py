#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_experiment.py

This file takes in the csv results of the experiment and makes a linechart out of it.

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
        here() / config["EXPERIMENT"]["INDIVIDUAL_EXPERIMENT"], index_col=0
    )
    mmd_name = "Maximum Mean Discrepancy"
    gnoise_name = "Injected Gaussian Noise to Coordinates"
    results.columns = [mmd_name, gnoise_name]

    results = results.set_index(gnoise_name)
    sns.lineplot(
        data=results,
        x=gnoise_name,
        y=mmd_name,
        # hue="coherence",
        # style="choice",
    )
    # add title
    plt.title(
        f"{mmd_name} computed from Weisfeiler-Lehman kernel vs {gnoise_name}."
    )
    print("Saving")
    plt.tight_layout()
    plt.savefig(
        config["EXPERIMENT"]["PLOT_PATH"] / Path("plot_experiment.png")
    )


if __name__ == "__main__":
    main()
