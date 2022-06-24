#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""esm_mutation.py

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from pyprojroot import here

N_RUNS = 10


def normalize(df):
    cols = [
        col for col in df.columns if "perturb" not in col and "run" not in col
    ]
    for col in cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def load_data():
    """Load data."""
    df = pd.DataFrame()
    for run in range(N_RUNS):
        df_run = pd.read_csv(
            here()
            / f"data/systematic/human/fixed_length_kernels/esm/mutation/{run}/mutation_mmds.csv"
        )
        df_run = df_run.assign(run=run)

        df = pd.concat([df, normalize(df_run),])
    return df


def annotate(data, **kws):
    # for i, eps_value in enumerate(data.eps_value.unique()):
    # eps_df = data[data["eps_value"] == eps_value]
    r_ps = list()
    r_ss = list()
    for run in data.run.unique():
        run_df = data[data["run"] == run]
        r_p, p_p = sp.stats.pearsonr(
            run_df["Mutation Probability"], run_df["Normalized MMD"]
        )
        r_ps.append(r_p)
        r_s, p_s = sp.stats.spearmanr(
            run_df["Mutation Probability"], run_df["Normalized MMD"]
        )
        r_ss.append(r_s)

    avg_rp = np.mean(r_ps)
    avg_rs = np.mean(r_ss)
    ax = plt.gca()
    ax.text(
        0.8 - 0.23 * 0,
        0.05,
        f"ESM"
        + f"\n"
        + r"$\rho_P=$"
        + f"{round(avg_rp, 2)}"
        + "\n"
        + r"$\rho_S=$"
        + f"{round(avg_rs, 2)}",
        # color=palette[i],
        transform=ax.transAxes,
        fontsize=8,
    )


def main():
    df = load_data()
    df = df.melt(id_vars=["run", "perturb"])
    df = df.rename(
        columns={
            "perturb": "Mutation Probability",
            "variable": "kernel",
            "value": "Normalized MMD",
        }
    )
    # filter out rows with sigma > 100
    df = df[
        (df["kernel"] != "sigma=100.0")
        & (df["kernel"] != "sigma=1000.0")
        & (df["kernel"] != "sigma=10000.0")
        & (df["kernel"] != "sigma=100000.0")
    ]

    df.reset_index(drop=True, inplace=True)
    df = df.assign(cst=1)
    g = sns.relplot(
        x="Mutation Probability",
        y="Normalized MMD",
        hue="kernel",
        col="kernel",
        data=df,
        kind="line",
        aspect=1,
        height=3,
        palette=sns.color_palette("mako_r", df.kernel.nunique()),
        ci=100,
        legend=False,
        col_wrap=3,
    )
    g.map_dataframe(annotate)
    titles = [
        r"RBF Kernel $\sigma=1\cdot 10^{-5}$",
        r"RBF Kernel $\sigma=0.0001$",
        r"RBF Kernel $\sigma=0.001$",
        r"RBF Kernel $\sigma=0.01$",
        r"RBF Kernel $\sigma=0.1$",
        r"RBF Kernel $\sigma=1$",
        r"RBF Kernel $\sigma=10$",
        r"Linear Kernel",
    ]
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title(titles[i])

    plt.savefig(here() / "exploring/systematic_analysis/res_5.pdf")


if __name__ == "__main__":
    main()
