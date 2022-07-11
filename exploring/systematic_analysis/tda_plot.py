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
from torch import normal

from proteinmetrics.utils.plots import setup_plotting_parameters

N_RUNS = 10


def normalize(df):
    cols = [col for col in df.columns if "run" not in col]
    for col in cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def load_data():
    """Load data."""
    df = pd.DataFrame()
    perturbations = ["gaussian_noise", "shear", "taper", "twist"]
    for perturbation in perturbations:
        df_perturb = pd.DataFrame()
        for run in range(N_RUNS):
            df_run = normalize(
                pd.read_csv(
                    here()
                    / f"data/systematic/human/tda/{perturbation}/{run}/{perturbation}_mmds.csv"
                )
            )
            df_run = df_run.assign(run=run)

            df_perturb = pd.concat([df_perturb, normalize(df_run),])
        df_perturb = df_perturb.assign(perturb_type=perturbation)
        df = pd.concat([df, df_perturb])
    return df


def annotate(data, **kws):
    for i, kernel in enumerate(data.Kernel.unique()):
        kernel_df = data[data["Kernel"] == kernel]
        r_ps = list()
        r_ss = list()
        for run in kernel_df.run.unique():
            run_df = kernel_df[kernel_df["run"] == run]
            r_p, p_p = sp.stats.pearsonr(
                run_df["Perturbation (%)"], run_df["Normalized MMD"]
            )
            r_ps.append(r_p)
            r_s, p_s = sp.stats.spearmanr(
                run_df["Perturbation (%)"], run_df["Normalized MMD"]
            )
            r_ss.append(r_s)

        avg_rp = np.mean(r_ps)
        avg_rs = np.mean(r_ss)
        ax = plt.gca()
        if kernel == "Persistence Fisher Kernel":
            ax.text(
                0.8 - 0.23 * i,
                0.05,
                f"PFK"
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
        else:
            ax.text(
                0.8 - 0.23 * i,
                0.05,
                f"MSK"
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
    setup_plotting_parameters()
    df = load_data()
    df = df.melt(id_vars=["run", "perturb", "perturb_type"])
    df = df.rename(
        columns={
            "perturb": "Perturbation (%)",
            "variable": "kernel",
            "value": "Normalized MMD",
        }
    )

    # Rename kernels
    df["kernel"] = df["kernel"].str.replace(
        "persistence_fisher_bandwidth=1;bandwidth_fisher=1",
        "Persistence Fisher Kernel",
    )
    df["kernel"] = df["kernel"].str.replace(
        "mutli_scale_kernel_bandwidth=1;bandwidth_fisher=1",
        "Multi-scale kernel",
    )

    df.reset_index(drop=True, inplace=True)
    df.rename(columns={"kernel": "Kernel"}, inplace=True)
    g = sns.relplot(
        x="Perturbation (%)",
        y="Normalized MMD",
        hue="Kernel",
        col="perturb_type",
        data=df,
        kind="line",
        aspect=1,
        height=3,
        palette=sns.color_palette("mako_r", df.Kernel.nunique()),
        ci=100,
        # legend=False,
        col_wrap=2,
    )
    g.map_dataframe(annotate)

    titles = ["Gaussian Noise", "Shear", "Taper", "Twist"]
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title(titles[i])

    plt.savefig(here() / "exploring/systematic_analysis/res_6.pdf")

    df.groupby(["Perturbation (%)", "perturb_type", "Kernel"]).std()[
        "Normalized MMD"
    ].reset_index().groupby(["perturb_type", "Kernel"]).mean().round(3)[
        "Normalized MMD"
    ].to_latex(
        here() / "exploring/systematic_analysis/res_6_std.tex"
    )


if __name__ == "__main__":
    main()
