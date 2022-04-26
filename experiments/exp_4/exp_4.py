#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""exp_4.py

The goal of this experiment is to investigate the variance of the data on MMD values across proteomes.

"""

import logging
import os
import random
from multiprocessing.sharedctypes import Value
from re import L

import hydra
import numpy as np
import pandas as pd
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here
from torch import embedding
from tqdm import tqdm

from proteinggnnmetrics.descriptors import ESM
from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.kernels import LinearKernel
from proteinggnnmetrics.loaders import list_pdb_files, load_graphs
from proteinggnnmetrics.paths import ECOLI_PROTEOME, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates, Sequence
from proteinggnnmetrics.perturbations import Mutation
from proteinggnnmetrics.utils.functions import flatten_lists

log = logging.getLogger(__name__)


def remove_fragments(files):
    return [file for file in files if "F1" in str(file)]


def variance_organism(organism, cfg):
    if organism == "ecoli":
        pdb_files = list_pdb_files(HUMAN_PROTEOME)
    elif organism == "human":
        pdb_files = list_pdb_files(ECOLI_PROTEOME)
    else:
        raise ValueError(
            f"Invalid organism."
            f"Should be one of {cfg.variables. data.organisms}"
        )

    log.info(f"Number of PDB files: {len(pdb_files)}")
    pdb_files = remove_fragments(pdb_files)
    log.info(f"Number of PDB files without fragments: {len(pdb_files)}")

    wl_epsilon_pipeline = pipeline.Pipeline(
        [
            (
                "coordinates",
                Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
            ),
            (
                "contact map",
                ContactMap(
                    metric="euclidean",
                    n_jobs=cfg.compute.n_jobs,
                ),
            ),
            (
                "epsilon graph",
                EpsilonGraph(epsilon=8, n_jobs=cfg.compute.n_jobs),
            ),
        ]
    )

    for size in cfg.experiments.size_range:
        log.info(f"Assessing variance with size: {size}")
        mmd_for_size = []
        for run in tqdm(range(cfg.experiments.n_runs_per_size)):
            sampled_files = random.Random(run).sample(pdb_files, size * 2)
            midpoint = int(size / 2)
            set_1 = wl_epsilon_pipeline.fit_transform(sampled_files[midpoint:])
            set_2 = wl_epsilon_pipeline.fit_transform(sampled_files[:midpoint])
            graphs_1 = load_graphs(set_1, graph_type="eps_graph")
            graphs_2 = load_graphs(set_2, graph_type="eps_graph")
            mmd = MaximumMeanDiscrepancy(
                biased=True,
                squared=True,
                kernel=WeisfeilerLehmanKernel(
                    n_jobs=cfg.compute.n_jobs,
                    n_iter=5,
                    normalize=True,
                    biased=True,
                ),  # type: ignore
            ).compute(graphs_1, graphs_2)
            mmd_for_size.append(mmd)
        mmd_for_size_df = pd.DataFrame(mmd_for_size)
        mmd_for_size_df["size"] = size
        mmd_for_size_df.to_csv(
            here()
            / cfg.experiments.results
            / f"mmd_for_size_{organism}_{size}.csv",
            index=False,
        )


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf_4")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    os.makedirs(here() / cfg.experiments.results, exist_ok=True)
    for organism in cfg.variables.data.organisms:
        variance_organism(organism, cfg)


if __name__ == "__main__":
    main()
