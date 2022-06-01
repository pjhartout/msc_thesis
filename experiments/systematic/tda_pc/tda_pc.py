#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tda.py

Topological Data Analysis (TDA) on point clouds (PC) perturbations.

"""

import argparse
import logging
import os
import random
import sys
from enum import unique
from multiprocessing.sharedctypes import Value
from pathlib import Path
from re import A
from tabnanny import verbose
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

from proteinmetrics.descriptors import (
    ESM,
    ClusteringHistogram,
    DegreeHistogram,
    DistanceHistogram,
    LaplacianSpectrum,
    RamachandranAngles,
    TopologicalDescriptor,
)
from proteinmetrics.distance import MaximumMeanDiscrepancy
from proteinmetrics.graphs import ContactMap, EpsilonGraph, KNNGraph
from proteinmetrics.kernels import LinearKernel, PersistenceFisherKernel
from proteinmetrics.loaders import list_pdb_files, load_descriptor
from proteinmetrics.paths import DATA_HOME, ECOLI_PROTEOME, HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates, Sequence
from proteinmetrics.perturbations import (
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
from proteinmetrics.protein import Protein
from proteinmetrics.utils.functions import (
    distribute_function,
    flatten_lists,
    make_dir,
    remove_fragments,
    save_obj,
)

log = logging.getLogger(__name__)


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


def pc_perturbation_worker(
    cfg, experiment_steps, perturbation, unperturbed, perturbed,
):
    experiment_steps_perturbed = experiment_steps[1:]
    experiment_steps_perturbed.insert(0, perturbation)
    perturbed = perturbed.copy()
    perturbed = pipeline.Pipeline(experiment_steps_perturbed).fit_transform(
        perturbed
    )
    log.info("Computed the representations.")

    log.info("Extracting graphs")

    perturbed_protein_names = idx2name2run(cfg, perturbed=True)
    unperturbed_protein_names = idx2name2run(cfg, perturbed=False)

    mmd_runs_n_iter = dict()
    for bandwidth in cfg.meta.kernels[0]["persistence_fisher"][0]["bandwidth"]:
        for bandwidth_fisher in cfg.meta.kernels[0]["persistence_fisher"][1][
            "bandwidth_fisher"
        ]:
            mmd_runs = list()
            for run in range(cfg.meta.n_runs):
                log.info(f"Run {run}")
                unperturbed_run = filter_protein_using_name(
                    unperturbed, unperturbed_protein_names[run].tolist()
                )
                perturbed_run = filter_protein_using_name(
                    perturbed, perturbed_protein_names[run].tolist()
                )

                unperturbed_descriptor_run = load_descriptor(
                    unperturbed_run,
                    graph_type="contact_graph",
                    descriptor="diagram",
                )
                perturbed_descriptor_run = load_descriptor(
                    perturbed_run,
                    graph_type="contact_graph",
                    descriptor="diagram",
                )

                log.info("Computing the kernel.")

                mmd = MaximumMeanDiscrepancy(
                    biased=False,
                    squared=True,
                    kernel=PersistenceFisherKernel(
                        bandwidth=bandwidth,
                        bandwidth_fisher=bandwidth_fisher,
                        n_jobs=cfg.compute.n_jobs,
                        verbose=cfg.debug.verbose,
                    ),  # type: ignore
                    verbose=cfg.debug.verbose,
                ).compute(unperturbed_descriptor_run, perturbed_descriptor_run)
                mmd_runs.append(mmd)

            mmd_runs_n_iter[
                f"bandwidth={bandwidth};bandwidth_fisher={bandwidth_fisher}"
            ] = mmd_runs

    return mmd_runs_n_iter


def save_mmd_experiment(cfg, mmds, perturbation_type):
    mmds = (
        pd.concat(mmds)
        .reset_index()
        .set_index(["index", "perturb"])
        .rename_axis(index={"index": "run"})
    )
    target_dir = (
        here()
        / cfg.paths.data
        / cfg.paths.systematic
        / cfg.paths.human
        / cfg.paths.tda
        / perturbation_type
    )
    make_dir(target_dir)

    mmds.to_csv(target_dir / f"{perturbation_type}_mmds.csv")

    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    log.info(f"WROTE FILE in {target_dir}")
    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


def twist_perturbation_wl_pc(
    cfg, perturbed, unperturbed, experiment_steps, **kwargs,
):
    log.info("Perturbing proteins with twist.")

    def twist_perturbation_worker(p_perturb, perturbed, unperturbed):
        log.info(f"Twist rad/Å set to {p_perturb}.")
        perturbation = (
            f"twist_{p_perturb}",
            Twist(
                alpha=p_perturb,
                random_state=hash(
                    str(perturbed)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = pc_perturbation_worker(
            cfg, experiment_steps, perturbation, unperturbed, perturbed,
        )
        mmd_df = pd.DataFrame(mmd_runs).assign(perturb=p_perturb)
        log.info(f"Computed the MMD with twist {p_perturb}.")
        return mmd_df

    mmds = distribute_function(
        func=twist_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.twist.min,
            cfg.perturbations.twist.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_perturb,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="Twisting experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
    )
    save_mmd_experiment(
        cfg, mmds, perturbation_type="twist",
    )


def shear_perturbation_wl_pc(
    cfg, perturbed, unperturbed, experiment_steps, **kwargs,
):
    log.info("Perturbing proteins with shear.")

    def twist_perturbation_worker(p_perturb, perturbed, unperturbed):
        log.info(f"Shear set to {p_perturb}.")
        perturbation = (
            f"shear_{p_perturb}",
            Shear(
                shear_x=p_perturb,
                shear_y=p_perturb,
                random_state=hash(
                    str(perturbed)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = pc_perturbation_worker(
            cfg,
            experiment_steps,
            perturbation,
            unperturbed,
            perturbed,
            graph_type,
        )
        mmd_df = pd.DataFrame(mmd_runs).assign(perturb=p_perturb)
        log.info(f"Computed the MMD with shearing {p_perturb}.")
        return mmd_df

    mmds = distribute_function(
        func=twist_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.shear.min,
            cfg.perturbations.shear.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_perturb,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="Shearing experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
    )
    save_mmd_experiment(
        cfg, mmds, perturbation_type="shear",
    )


def taper_perturbation_wl_pc(
    cfg, perturbed, unperturbed, experiment_steps, **kwargs,
):
    log.info("Perturbing proteins with taper.")

    def twist_perturbation_worker(p_perturb, perturbed, unperturbed):
        log.info(f"taper set to {p_perturb}.")
        perturbation = (
            f"taper_{p_perturb}",
            Taper(
                a=p_perturb,
                b=p_perturb,
                random_state=hash(
                    str(perturbed)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = pc_perturbation_worker(
            cfg,
            experiment_steps,
            perturbation,
            unperturbed,
            perturbed,
            graph_type,
        )
        mmd_df = pd.DataFrame(mmd_runs).assign(perturb=p_perturb)
        log.info(f"Computed the MMD with tapering {p_perturb}.")
        return mmd_df

    mmds = distribute_function(
        func=twist_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.taper.min,
            cfg.perturbations.taper.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_perturb,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="tapering experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
    )
    save_mmd_experiment(
        cfg, mmds, perturbation_type="taper",
    )


def gaussian_perturbation_wl_pc(
    cfg, perturbed, unperturbed, experiment_steps, **kwargs,
):
    log.info("Perturbing proteins with gaussian.")

    def twist_perturbation_worker(p_perturb, perturbed, unperturbed):
        log.info(f"gaussian set to {p_perturb}.")
        perturbation = (
            f"gaussian_{p_perturb}",
            GaussianNoise(
                noise_mean=0,
                noise_std=p_perturb,
                random_state=hash(
                    str(perturbed)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = pc_perturbation_worker(
            cfg,
            experiment_steps,
            perturbation,
            unperturbed,
            perturbed,
            graph_type,
        )
        mmd_df = pd.DataFrame(mmd_runs).assign(perturb=p_perturb)
        log.info(f"Computed the MMD with gaussian noise {p_perturb}.")
        return mmd_df

    mmds = distribute_function(
        func=twist_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.gaussian_noise.min,
            cfg.perturbations.gaussian_noise.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_perturb,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="gaussian noise experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
    )
    save_mmd_experiment(
        cfg, mmds, perturbation_type="gaussian_noise",
    )


def tda_experiment_pc_perturbation(
    cfg: DictConfig, perturbation: str,
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
        (
            "tda",
            TopologicalDescriptor(
                tda_descriptor_type="diagram",
                epsilon=1,
                n_bins=100,
                use_caching=False,
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
    ]

    unperturbed = load_proteins_from_config(cfg, perturbed=False)
    unperturbed = pipeline.Pipeline(base_feature_steps).fit_transform(
        unperturbed
    )
    perturbed = load_proteins_from_config(cfg, perturbed=True)
    perturbed = pipeline.Pipeline([base_feature_steps[0]]).fit_transform(
        perturbed
    )

    if perturbation == "twist":
        log.info("Compute twist")
        twist_perturbation_wl_pc(
            cfg, perturbed, unperturbed, base_feature_steps,
        )
    elif perturbation == "shear":
        log.info("Compute shear")
        shear_perturbation_wl_pc(
            cfg, perturbed, unperturbed, base_feature_steps,
        )
    elif perturbation == "taper":
        log.info("Compute taper")
        taper_perturbation_wl_pc(
            cfg, perturbed, unperturbed, base_feature_steps,
        )
    elif perturbation == "gaussian_noise":
        log.info("Compute gaussian noise")
        gaussian_perturbation_wl_pc(
            cfg, perturbed, unperturbed, base_feature_steps,
        )

    else:
        raise ValueError("Invalid perturbation")


@hydra.main(
    version_base=None,
    config_path=str(here()) + "/conf/",
    config_name="systematic",
)
def main(cfg: DictConfig):
    log.info("Starting graph_experiments.py")
    log.info("Running with config:")
    log.info(OmegaConf.to_yaml(cfg))

    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    log.info("DATA_DIR")
    log.info(here())
    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Start with Weisfeiler-Lehman-based-experiments.
    # outside for loops for n_iters and k.

    tda_experiment_pc_perturbation(
        cfg=cfg, perturbation="twist",
    )


if __name__ == "__main__":
    main()
