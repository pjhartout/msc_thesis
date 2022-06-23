#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""esm_mutation.py

ESM on mutation

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
from sklearn.metrics import pairwise_distances
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
from proteinmetrics.kernels import (
    GaussianKernel,
    LinearKernel,
    PersistenceFisherKernel,
)
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


def load_proteins_from_config_run(
    cfg: DictConfig, perturbed: bool, run: int
) -> List[Path]:
    """Loads unique unperturbed protein file names"""
    protein_sets = list()
    if perturbed:
        protein_sets.append(
            pd.read_csv(
                here()
                / cfg.meta.splits_dir
                / f"data_split_{run}_perturbed.csv",
                sep="\t",
            )
        )
    else:
        protein_sets.append(
            pd.read_csv(
                here()
                / cfg.meta.splits_dir
                / f"data_split_{run}_unperturbed.csv",
                sep="\t",
            )
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


def idx2name2run(cfg: DictConfig, perturbed: bool, run) -> pd.DataFrame:
    """Loads unique unperturbed protein file names

    Args:
        cfg (DictConfig): Configuration.

    Returns:
        pd.DataFrame: dataframe where each column contains the names of proteins to use in one run.
    """
    protein_sets = []
    # for file in os.listdir(here() / cfg.meta.splits_dir):
    if perturbed:
        protein_sets.append(
            pd.read_csv(
                here()
                / cfg.meta.splits_dir
                / f"data_split_{run}_perturbed.csv",
                sep="\t",
            )
        )
    else:
        protein_sets.append(
            pd.read_csv(
                here()
                / cfg.meta.splits_dir
                / f"data_split_{run}_perturbed.csv",
                sep="\t",
            )
        )
    return (
        pd.concat(protein_sets, axis=1)
        .set_axis(list(range(cfg.meta.n_runs)), axis=1, inplace=False)
        .applymap(lambda x: x.split(".")[0])
    )


def filter_protein_using_name(protein, protein_names):
    return [protein for protein in protein if protein.name in protein_names]


def pc_perturbation_worker(
    cfg, experiment_steps, perturbation, unperturbed, perturbed, run
):
    experiment_steps_perturbed = experiment_steps[1:]
    experiment_steps_perturbed.insert(0, perturbation)
    perturbed = pipeline.Pipeline(experiment_steps_perturbed).fit_transform(
        perturbed
    )

    perturbed_protein_names = idx2name2run(cfg, perturbed=True, run=run)
    unperturbed_protein_names = idx2name2run(cfg, perturbed=False, run=run)

    # For every run - compute x-y
    pre_computed_products = list()
    log.info(f"Run {run}")
    unperturbed_run = filter_protein_using_name(
        unperturbed, unperturbed_protein_names[run].tolist()
    )
    perturbed_run = filter_protein_using_name(
        perturbed, perturbed_protein_names[run].tolist()
    )

    unperturbed_descriptor_run = np.asarray(
        [protein.embeddings["esm"] for protein in unperturbed_run]
    )
    perturbed_descriptor_run = np.asarray(
        [protein.embeddings["esm"] for protein in perturbed_run]
    )

    products = {
        # The np.ones is used here because
        # exp(sigma*(x-x)**2) = 1(n x n)
        "K_XX": pairwise_distances(
            unperturbed_descriptor_run,
            unperturbed_descriptor_run,
        ),
        "K_YY": pairwise_distances(
            perturbed_descriptor_run,
            perturbed_descriptor_run,
            metric="euclidean",
        ),
        "K_XY": pairwise_distances(
            unperturbed_descriptor_run,
            perturbed_descriptor_run,
            metric="euclidean",
        ),
    }
    pre_computed_products.append(products)
    # pre_computed_products[f"sigma={sigma}"] = pre_computed_products_sigma

    # For every run and for every sigma - compute gaussian kernel
    mmd_runs_sigma = dict()
    for sigma in cfg.meta.kernels[1]["gaussian"][0]["bandwidth"]:
        mmd_runs = list()
        log.info(f"Run {run}")
        log.info("Computing the kernel.")

        kernel = GaussianKernel(sigma=sigma, pre_computed_product=True)

        mmd = MaximumMeanDiscrepancy(
            biased=False,
            squared=True,
            verbose=cfg.debug.verbose,
            kernel=kernel,
        ).compute(
            kernel.compute_matrix(pre_computed_products[run]["K_XX"]),
            kernel.compute_matrix(pre_computed_products[run]["K_YY"]),
            kernel.compute_matrix(pre_computed_products[run]["K_XY"]),
        )
        mmd_runs.append(mmd)
        mmd_runs_sigma[f"sigma={sigma}"] = mmd_runs

    # For every run - compute linear kernel
    mmd_linear_kernel_runs = []
    log.info(f"Run {run}")
    unperturbed_run = filter_protein_using_name(
        unperturbed, unperturbed_protein_names[run].tolist()
    )
    perturbed_run = filter_protein_using_name(
        perturbed, perturbed_protein_names[run].tolist()
    )
    unperturbed_descriptor_run = np.asarray(
        [protein.embeddings["esm"] for protein in unperturbed_run]
    )
    perturbed_descriptor_run = np.asarray(
        [protein.embeddings["esm"] for protein in perturbed_run]
    )

    log.info("Computing the kernel.")

    mmd = MaximumMeanDiscrepancy(
        biased=False,
        squared=True,
        kernel=LinearKernel(
            n_jobs=cfg.compute.n_jobs,
            normalize=False,
        ),  # type: ignore
        verbose=cfg.debug.verbose,
    ).compute(unperturbed_descriptor_run, perturbed_descriptor_run)
    mmd_linear_kernel_runs.append(mmd)

    # We make linear kernel part of the same dict to simplify things
    mmd_runs_sigma["linear_kernel"] = mmd_linear_kernel_runs
    return mmd_runs_sigma


def save_mmd_experiment(cfg, mmds, perturbation_type, run):
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
        / cfg.paths.esm
        / perturbation_type
        / run
    )
    make_dir(target_dir)

    mmds.to_csv(target_dir / f"{perturbation_type}_mmds.csv")

    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    log.info(f"WROTE FILE in {target_dir}")
    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


def get_longest_protein_dummy_sequence(sampled_files, cfg: DictConfig) -> int:
    seq = Sequence(n_jobs=cfg.compute.n_jobs)
    sequences = seq.fit_transform(sampled_files)
    longest_sequence = max([len(protein.sequence) for protein in sequences])
    return longest_sequence


def mutation_perturbation_esm(
    cfg,
    perturbed,
    unperturbed,
    experiment_steps,
    run,
    **kwargs,
):
    log.info("Perturbing proteins with mutations.")

    def mutation_perturbation_worker(p_perturb, perturbed, unperturbed):
        log.info(f"gaussian set to {p_perturb}.")
        perturbation = (
            f"gaussian_{p_perturb}",
            Mutation(
                p_mutate=p_perturb,
                random_state=np.random.RandomState(
                    divmod(hash(str(perturbed)), 42)[1]
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = pc_perturbation_worker(
            cfg, experiment_steps, perturbation, unperturbed, perturbed, run
        )
        mmd_df = pd.DataFrame(mmd_runs).assign(perturb=p_perturb)
        log.info(f"Computed the MMD with mutation {p_perturb}.")
        return mmd_df

    mmds = distribute_function(
        func=mutation_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.mutation.min,
            cfg.perturbations.mutation.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_perturb,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="mutation experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
    )
    save_mmd_experiment(cfg, mmds, perturbation_type="mutation", run=run)


def tda_experiment_pc_perturbation(
    cfg: DictConfig,
    perturbation: str,
):

    unperturbed = load_proteins_from_config(cfg, perturbed=False)
    perturbed = load_proteins_from_config(cfg, perturbed=True)

    longest_unperturbed = get_longest_protein_dummy_sequence(unperturbed, cfg)
    longest_perturbed = get_longest_protein_dummy_sequence(perturbed, cfg)
    longest_sequence = max([longest_unperturbed, longest_perturbed])

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
            "esm",
            ESM(
                size="M",
                longest_sequence=longest_sequence,
                n_jobs=cfg.compute.n_jobs,
                verbose=True,
            ),
        ),
    ]
    for run in range(cfg.meta.n_runs):
        unperturbed = load_proteins_from_config_run(
            cfg, perturbed=False, run=run
        )
        perturbed = load_proteins_from_config_run(cfg, perturbed=True, run=run)
        unperturbed = pipeline.Pipeline(base_feature_steps).fit_transform(
            unperturbed
        )
        perturbed = pipeline.Pipeline([base_feature_steps[0]]).fit_transform(
            perturbed
        )
        if perturbation == "mutation":
            log.info("Compute mutation")
            mutation_perturbation_esm(
                cfg, perturbed, unperturbed, base_feature_steps, run
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
        cfg=cfg,
        perturbation="mutation",
    )


if __name__ == "__main__":
    main()
