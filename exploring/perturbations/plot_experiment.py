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


def main():

    # Load the data
    results = pd.read_csv(
        here() / config["EXPERIMENT"]["EXPERIMENT_REL_PATH"], index_col=0
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
        f"{mmd_name} computed from Weisfeiler-Lehman kernel vs {gnoise_name}  \n on two different but overlapping fragments of the same protein."
    )
    plt.show()


if __name__ == "__main__":
    main()
