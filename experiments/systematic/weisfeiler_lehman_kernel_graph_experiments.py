#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""graph_experiments.py

The idea is to deal with all graph experiments here.
Steps:
    1. Compute unperturbed set
        1. Point cloud
    2. Generate graphs
        1.
    3. Compute MMD

Structure of output:
/data/systematic/wl_experiments/

"""

import logging
import os
import random
from enum import unique
from pathlib import Path
from tkinter import E
from typing import Any, Dict, List, Tuple, Union

import hydra
import numpy as np
import pandas as pd
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here
from tqdm import tqdm

from proteinggnnmetrics.descriptors import (
    ESM,
    ClusteringHistogram,
    DegreeHistogram,
    DistanceHistogram,
    LaplacianSpectrum,
    RamachandranAngles,
    TopologicalDescriptor,
)
from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph, KNNGraph
from proteinggnnmetrics.kernels import LinearKernel
from proteinggnnmetrics.loaders import list_pdb_files, load_graphs
from proteinggnnmetrics.paths import DATA_HOME, ECOLI_PROTEOME, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates, Sequence
from proteinggnnmetrics.perturbations import (
    AddConnectedNodes,
    AddEdges,
    GaussianNoise,
    GraphPerturbation,
    Mutation,
    RemoveEdges,
    RewireEdges,
    Shear,
    Taper,
    Twist,
)
from proteinggnnmetrics.protein import Protein
from proteinggnnmetrics.utils.functions import (
    distribute_function,
    flatten_lists,
    make_dir,
    remove_fragments,
    save_obj,
)

log = logging.getLogger(__name__)

N_JOBS = 4


def load_proteins_from_config(cfg: DictConfig, perturbed: bool) -> List[Path]:
    """Loads unique unperturbed protein file names

    Args:
        cfg (DictConfig): Configuration.

    Returns:
        List[Path]: unique proteins seen across runs
    """
    protein_sets = []
    for file in os.listdir(here() / cfg.meta.splits_dir):
        if perturbed:
            if "_perturbed" in file:
                protein_sets.append(
                    pd.read_csv(here() / cfg.meta.splits_dir / file, sep="\t")
                )
        else:
            if "_unperturbed" in file:
                protein_sets.append(
                    pd.read_csv(here() / cfg.meta.splits_dir / file, sep="\t")
                )
    unique_protein_fnames = pd.concat(protein_sets).drop_duplicates()
    unique_protein_fnames = [
        HUMAN_PROTEOME / fname
        for fname in unique_protein_fnames.pdb_id.tolist()
    ]
    log.info(
        f"Found {len(unique_protein_fnames)} unique proteins in each set."
    )

    return unique_protein_fnames  # type: ignore


def idx2name2run(cfg: DictConfig, perturbed: bool) -> pd.DataFrame:
    """Loads unique unperturbed protein file names

    Args:
        cfg (DictConfig): Configuration.

    Returns:
        pd.DataFrame: dataframe where each column contains the names of proteins to use in one run.
    """
    protein_sets = []
    for file in os.listdir(here() / cfg.meta.splits_dir):
        if perturbed:
            if "_perturbed" in file:
                protein_sets.append(
                    pd.read_csv(here() / cfg.meta.splits_dir / file, sep="\t")
                )
        else:
            if "_unperturbed" in file:
                protein_sets.append(
                    pd.read_csv(here() / cfg.meta.splits_dir / file, sep="\t")
                )
    return (
        pd.concat(protein_sets, axis=1)
        .set_axis(list(range(cfg.meta.n_runs)), axis=1, inplace=False)
        .applymap(lambda x: x.split(".")[0])
    )


def filter_protein_using_name(protein, protein_names):
    return [protein for protein in protein if protein.name in protein_names]


def twist_perturbation_wl_pc(
    cfg, perturbed, unperturbed, experiment_steps, graph_type, n_iter, **kwargs
):
    log.info("Perturbing proteins with twist.")

    def twist_perturbation_worker(alpha, perturbed, unperturbed):
        log.info(f"Twist rad/Å set to {alpha}.")
        perturbation = (
            f"twist_{alpha}",
            Twist(
                alpha=alpha,
                random_state=hash(
                    str(perturbed)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        experiment_steps_perturbed = experiment_steps[1:]
        experiment_steps_perturbed.insert(0, perturbation)
        perturbed = pipeline.Pipeline(
            experiment_steps_perturbed
        ).fit_transform(perturbed)
        log.info("Computed the representations.")

        log.info("Extracting graphs")

        perturbed_protein_names = idx2name2run(cfg, perturbed=True)
        unperturbed_protein_names = idx2name2run(cfg, perturbed=True)

        mmd_runs = []
        for run in range(2):
            log.info(f"Run {run}")
            unperturbed_run = filter_protein_using_name(
                unperturbed, unperturbed_protein_names[run].tolist()
            )
            perturbed_run = filter_protein_using_name(
                perturbed, perturbed_protein_names[run].tolist()
            )

            unperturbed_graphs = load_graphs(
                unperturbed_run, graph_type=graph_type
            )
            perturbed_graphs = load_graphs(
                perturbed_run, graph_type=graph_type
            )

            if cfg.debug.reduce_data:
                unperturbed_graphs = unperturbed_graphs[
                    : cfg.debug.sample_data_size
                ]
                perturbed_graphs = perturbed_graphs[
                    : cfg.debug.sample_data_size
                ]

            log.info("Computing the kernel.")

            mmd = MaximumMeanDiscrepancy(
                biased=True,
                squared=True,
                kernel=WeisfeilerLehmanKernel(
                    n_jobs=cfg.compute.n_jobs,
                    n_iter=n_iter,
                    normalize=True,
                    biased=True,
                ),  # type: ignore
            ).compute(unperturbed_graphs, perturbed_graphs)
            mmd_runs.append(mmd)
        mmd_pack = {"alpha": alpha, "mmd": mmd_runs}
        log.info(f"Computed the MMD with twist {alpha}.")
        return mmd_pack

    mmds = distribute_function(
        func=twist_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.twist.min,
            cfg.perturbations.twist.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_runs,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="Twisting experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
    )
    return mmds


def weisfeiler_lehman_experiment_pc_perturbation(
    cfg: DictConfig, graph_type: str, graph_extraction_param: int, n_iter: int,
):
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(
                granularity="CA", n_jobs=cfg.compute.n_jobs, verbose=True
            ),
        ),
        (
            "contact map",
            ContactMap(
                metric="euclidean", n_jobs=cfg.compute.n_jobs, verbose=True
            ),
        ),
    ]

    if graph_type == "knn":
        base_feature_steps.append(
            (
                "knn graph",
                KNNGraph(
                    n_jobs=cfg.compute.n_jobs,
                    n_neighbors=graph_extraction_param,
                    verbose=cfg.debug.verbose,
                ),
            )
        )
    elif graph_type == "eps_graph":
        base_feature_steps.append(
            (
                "epsilon graph",
                EpsilonGraph(
                    n_jobs=cfg.compute.n_jobs,
                    epsilon=graph_extraction_param,
                    verbose=cfg.debug.verbose,
                ),
            )
        )
    else:
        raise ValueError(f"Unknown graph type {graph_type}")

    unperturbed = load_proteins_from_config(cfg, perturbed=False)
    unperturbed = pipeline.Pipeline(base_feature_steps).fit_transform(
        unperturbed
    )
    perturbed = load_proteins_from_config(cfg, perturbed=True)
    perturbed = pipeline.Pipeline([base_feature_steps[0]]).fit_transform(
        perturbed
    )

    log.info("Compute twist")
    twist_mmds = twist_perturbation_wl_pc(
        cfg, perturbed, unperturbed, base_feature_steps, graph_type, n_iter
    )
    twist_mmds = (
        pd.DataFrame(twist_mmds)
        .explode(column="mmd")
        .reset_index()
        .set_index(["index", "alpha"])
        .rename_axis(index={"index": "run"})
    )

    target_dir = (
        DATA_HOME
        / cfg.paths.systematic
        / cfg.paths.human
        / cfg.paths.weisfeiler_lehman
        / graph_type
        / str(graph_extraction_param)
        / "twist"
    )
    make_dir(target_dir)
    twist_mmds.to_csv(target_dir / f"twist_mmds_n_iters_{n_iter}.csv")

    log.info(
        f"Done with {graph_type} {graph_extraction_param} with W-L config {n_iter}"
    )


@hydra.main(config_path=str(here()) + "/conf/", config_name="systematic")
def main(cfg: DictConfig):
    log.info("Starting graph_experiments.py")
    log.info("Running with config:")
    log.info(OmegaConf.to_yaml(cfg))

    # Start with Weisfeiler-Lehman-based-experiments.
    # outside for loops for n_iters and k.
    for n_iters in cfg.meta.kernels[3]["weisfeiler-lehman"][0]["n_iter"]:
        for k in cfg.meta.representations[1]["knn_graphs"]:
            weisfeiler_lehman_experiment_pc_perturbation(
                cfg=cfg,
                graph_type="eps_graph",
                graph_extraction_param=k,
                n_iter=n_iters,
            )

    # Epsilon experiments
    # KNN experiments


if __name__ == "__main__":
    main()
