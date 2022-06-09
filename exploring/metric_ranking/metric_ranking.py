#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""metric_ranking.py

Given an mmd config, a set of perturbations with multiple runs, can we check which metric performs best by checking the corr coef times the standard dev??

formula:
meta_metric = (1-corr_coef) * std_dev

"""

import json
import logging
import os
from pathlib import Path

import hydra
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here

from proteinmetrics.utils.functions import make_dir

log = logging.getLogger(__name__)


def get_paths_for_mmd_config(cfg):
    path = (
        here()
        / cfg.paths.data
        / cfg.paths.systematic
        / cfg.paths.human
        / cfg.paths.fixed_length_kernels
    )
    mmd_configs = {}
    for root, dirs, files in os.walk(path, topdown=False):
        if files != [] and ".csv" in files[0]:
            path = root + "/" + files[0]
            parsed_path = str(path).split("/")
            descriptor = parsed_path[-2]
            perturbation = parsed_path[-3]
            extraction_param = parsed_path[-4]
            representation = parsed_path[-5]
            mmd_config = f"{representation}_{extraction_param}_{descriptor}"
            if mmd_config not in mmd_configs.keys():
                mmd_configs[mmd_config] = []
            else:
                mmd_configs[mmd_config].append(path)
    df = pd.DataFrame.from_dict(mmd_configs, orient="index").T
    return df


def get_mmd_config_data(df_path):
    df = pd.DataFrame()
    for mmd_config in df_path.columns:
        mmd_config_df = pd.DataFrame()
        for idx, path in df_path[mmd_config].iteritems():
            if path is not None:
                mmd_config_df = pd.concat(
                    [
                        mmd_config_df,
                        pd.read_csv(path).assign(
                            perturb_type=str(path).split("/")[-3]
                        ),
                    ]
                )
        mmd_config_df = mmd_config_df.assign(mmd_config=mmd_config,)
        df = pd.concat([df, mmd_config_df])
    return df


def compute_corr_coef_for_mmd_config(df):
    data_cols = [
        col for col in df.columns if "sigma" in col or "linear" in col
    ]
    perturb_types = df.perturb_type.unique()
    df_coefs = pd.DataFrame()
    for config in df.mmd_config.unique():
        for perturb_type in perturb_types:
            df_config = df[(df.mmd_config == config)]
            for perturb_type in df.perturb_type.unique():
                df_config_perturb = df[(df.perturb_type == perturb_type)]
                mean_config = (
                    df_config_perturb.groupby(["perturb"]).mean().reset_index()
                )
                for data_col in data_cols:
                    corr_coef = mean_config["perturb"].corr(
                        mean_config[data_col], method="spearman"
                    )
                    # Check if corr_coef is nan
                    if np.isnan(corr_coef):
                        print(
                            f"Corr coef is nan for {config} {perturb_type} {data_col}"
                        )

                    df_coefs = pd.concat(
                        [
                            df_coefs,
                            pd.DataFrame(
                                {
                                    "config": config,
                                    "data_col": data_col,
                                    "perturb_type": perturb_type,
                                    "corr_coef": corr_coef,
                                },
                                index=[0],
                            ),
                        ],
                    )
    return df_coefs.groupby(["config", "data_col"]).mean()


def compute_avg_std_for_mmd_config(df):
    data_cols = [
        col for col in df.columns if "sigma" in col or "linear" in col
    ]
    perturb_types = df.perturb_type.unique()
    df_stds = pd.DataFrame()
    for config in df.mmd_config.unique():
        for perturb_type in perturb_types:
            df_config = df[(df.mmd_config == config)]
            for perturb_type in df.perturb_type.unique():
                df_config_perturb = df[(df.perturb_type == perturb_type)]
                std_config = (
                    df_config_perturb.groupby(["perturb"])
                    .std()
                    .mean()
                    .drop("run")
                )
                for data_col in data_cols:
                    df_stds = pd.concat(
                        [
                            df_stds,
                            pd.DataFrame(
                                {
                                    "config": config,
                                    "data_col": data_col,
                                    "perturb_type": perturb_type,
                                    "std": std_config[data_col],
                                },
                                index=[0],
                            ),
                        ],
                    )
    return df_stds.groupby(["config", "data_col"])["std"].mean()


def normalize_data(df):
    data_cols = [
        col for col in df.columns if "sigma" in col or "linear" in col
    ]
    for col in data_cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def compute_mmd_config_quality(
    corr_coef_df: pd.DataFrame, std_df: pd.DataFrame
) -> pd.DataFrame:
    """The idea here is to make a quality score for each mmd_config.
    The score is given by (1-corr_coef) + std, the lower the better.

    Args:
        corr_coef_df (pd.DataFrame): _description_
        std_df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    one_minus_corr_coef_df = 1 - corr_coef_df
    quality_df = one_minus_corr_coef_df.join(std_df)
    # TODO: do maybe a weighted sum to give more important to one vs. the other?
    quality_df["combined_score"] = one_minus_corr_coef_df["corr_coef"] + std_df
    return quality_df


@hydra.main(
    version_base=None,
    config_path=str(here()) + "/conf/",
    config_name="systematic",
)
def main(cfg: DictConfig):
    df_path = get_paths_for_mmd_config(cfg)
    df = get_mmd_config_data(df_path)
    df_normalized = normalize_data(df)
    corr_coef_df = compute_corr_coef_for_mmd_config(df_normalized)
    std_df = compute_avg_std_for_mmd_config(df_normalized)
    df_quality = compute_mmd_config_quality(corr_coef_df, std_df)
    df_quality.sort_values(by=["combined_score", "corr_coef", "std"]).to_csv(
        here() / cfg.paths.data / "metric_scores.csv"
    )


if __name__ == "__main__":
    main()
