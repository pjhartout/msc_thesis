#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""esm_mutation.py

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyprojroot import here


def normalize(df):
    cols = [col for col in df.columns if "perturb" not in col]
    for col in cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def load_data():
    """Load data."""
    df = normalize(
        pd.read_csv(
            here()
            / "data/systematic/human/fixed_length_kernels/esm/mutation/mutation_mmds.csv"
        )
    )
    return df


def main():
    df = load_data()
    df = df.rename(
        columns={
            "perturb": "Mutation Probability",
            "sigma=0.01": "Normalized MMD",
        }
    )
    # Make lineplot with seaborn
    palette = sns.color_palette("mako_r", 1)
    sns.lineplot(
        x="Mutation Probability", y="Normalized MMD", data=df, palette=palette
    )

    plt.savefig(here() / "exploring/systematic_analysis/res_5.pdf")


if __name__ == "__main__":
    main()
