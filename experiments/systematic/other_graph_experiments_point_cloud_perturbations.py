#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""other_graph_experiments.py

"""

import logging
import os
import random
from enum import unique
from pathlib import Path
from re import A
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
from proteinmetrics.kernels import LinearKernel
from proteinmetrics.loaders import list_pdb_files, load_descriptor, load_graphs
from proteinmetrics.paths import DATA_HOME, ECOLI_PROTEOME, HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates, Sequence
from proteinmetrics.perturbations import GaussianNoise, Shear, Taper, Twist
from proteinmetrics.protein import Protein
from proteinmetrics.utils.functions import distribute_function, make_dir

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


def point_cloud_perturbation_worker(
    cfg,
    experiment_steps,
    perturbation,
    unperturbed,
    perturbed,
    graph_type,
    descriptor,
):
    experiment_steps_perturbed = experiment_steps[1:]
    experiment_steps_perturbed.insert(0, perturbation)
    perturbed = pipeline.Pipeline(experiment_steps_perturbed).fit_transform(
        perturbed
    )

    log.info("Computed the representations.")

    log.info("Extracting graphs")

    perturbed_protein_names = idx2name2run(cfg, perturbed=True)
    unperturbed_protein_names = idx2name2run(cfg, perturbed=False)

    mmd_runs = []
    for run in range(cfg.meta.n_runs):
        log.info(f"Run {run}")
        unperturbed_run = filter_protein_using_name(
            unperturbed, unperturbed_protein_names[run].tolist()
        )
        perturbed_run = filter_protein_using_name(
            perturbed, perturbed_protein_names[run].tolist()
        )

        unperturbed_descriptor = load_descriptor(
            unperturbed_run, graph_type=graph_type, descriptor=descriptor
        )
        perturbed_descriptor = load_descriptor(
            perturbed_run, graph_type=graph_type, descriptor=descriptor
        )

        if cfg.debug.reduce_data:
            unperturbed_graphs = unperturbed_descriptor[
                : cfg.debug.sample_data_size
            ]
            perturbed_descriptor = perturbed_descriptor[
                : cfg.debug.sample_data_size
            ]

        log.info("Computing the kernel.")

        mmd = MaximumMeanDiscrepancy(
            biased=False,
            squared=True,
            kernel=LinearKernel(
                n_jobs=cfg.compute.n_jobs, normalize=True,
            ),  # type: ignore
        ).compute(unperturbed_descriptor, perturbed_descriptor)
        mmd_runs.append(mmd)
    return mmd_runs


def save_mmd_experiment(
    cfg,
    mmds,
    graph_type,
    graph_extraction_param,
    perturbation_type,
    descriptor: str,
    kernel_params: Union[None, Dict] = None,
):
    mmds = (
        pd.DataFrame(mmds)
        .explode(column="mmd")
        .reset_index()
        .set_index(["index", "perturb"])
        .rename_axis(index={"index": "run"})
    )
    target_dir = (
        DATA_HOME
        / cfg.paths.systematic
        / cfg.paths.human
        / cfg.paths.weisfeiler_lehman
        / graph_type
        / str(graph_extraction_param)
        / perturbation_type
        / descriptor
    )
    make_dir(target_dir)
    if kernel_params is None:
        mmds.to_csv(target_dir / f"{perturbation_type}_mmds.csv")
    else:
        kernel_spec_string = ""
        for k, v in kernel_params.items():
            kernel_spec_string += f"{k}={v}_"
        mmds.to_csv(
            target_dir / f"{perturbation_type}_{kernel_spec_string}mmds.csv"
        )
    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    log.info(f"WROTE FILE in {target_dir}")
    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


def twist_perturbation_linear_kernel(
    cfg,
    perturbed,
    unperturbed,
    experiment_steps,
    graph_type,
    graph_extraction_param,
    descriptor,
    **kwargs,
):
    log.info("Perturbing proteins with twist.")

    def remove_edges_perturbation_worker(
        perturb, perturbed, unperturbed, graph_type
    ):
        log.info(f"Pertubation set to {perturb}.")
        perturbation = (
            f"twist_{perturb}",
            Twist(
                alpha=perturb,
                random_state=hash(
                    str(perturbed)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = point_cloud_perturbation_worker(
            cfg,
            experiment_steps,
            perturbation,
            unperturbed,
            perturbed,
            graph_type,
            descriptor,
        )
        mmd_pack = {"perturb": perturb, "mmd": mmd_runs}

        return mmd_pack

    mmds = distribute_function(
        func=remove_edges_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.twist.min,
            cfg.perturbations.twist.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_perturb,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="Twist experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
        graph_type=graph_type,
        descriptor=descriptor,
    )

    save_mmd_experiment(
        cfg,
        mmds,
        graph_type,
        graph_extraction_param,
        "twist",
        descriptor,
        kernel_params=None,
    )


def shear_perturbation_linear_kernel(
    cfg,
    perturbed,
    unperturbed,
    experiment_steps,
    graph_type,
    graph_extraction_param,
    descriptor,
    **kwargs,
):
    log.info("Perturbing proteins with shear.")

    def shear_perturbation_worker(
        perturb, perturbed, unperturbed, graph_type, descriptor
    ):
        log.info(f"Pertubation set to {perturb}.")
        perturbation = (
            f"shear_{perturb}",
            Shear(
                shear_x=perturb,
                shear_y=perturb,
                random_state=hash(
                    str(perturbed)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = point_cloud_perturbation_worker(
            cfg,
            experiment_steps,
            perturbation,
            unperturbed,
            perturbed,
            graph_type,
            descriptor,
        )
        mmd_pack = {"perturb": perturb, "mmd": mmd_runs}
        return mmd_pack

    mmds = distribute_function(
        func=shear_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.shear.min,
            cfg.perturbations.shear.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_perturb,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="Shear experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
        graph_type=graph_type,
        descriptor=descriptor,
    )

    save_mmd_experiment(
        cfg,
        mmds,
        graph_type,
        graph_extraction_param,
        "shear",
        descriptor,
        kernel_params=None,
    )


def taper_perturbation_linear_kernel(
    cfg,
    perturbed,
    unperturbed,
    experiment_steps,
    graph_type,
    graph_extraction_param,
    descriptor,
    **kwargs,
):
    log.info("Perturbing proteins with taper.")

    def taper_perturbation_worker(
        perturb, perturbed, unperturbed, graph_type, descriptor
    ):
        log.info(f"Pertubation set to {perturb}.")
        perturbation = (
            f"taper_{perturb}",
            Taper(
                a=perturb,
                b=perturb,
                random_state=hash(
                    str(perturbed)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = point_cloud_perturbation_worker(
            cfg,
            experiment_steps,
            perturbation,
            unperturbed,
            perturbed,
            graph_type,
            descriptor,
        )
        mmd_pack = {"perturb": perturb, "mmd": mmd_runs}

        return mmd_pack

    mmds = distribute_function(
        func=taper_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.taper.min,
            cfg.perturbations.taper.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_perturb,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="Taper experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
        graph_type=graph_type,
        descriptor=descriptor,
    )

    save_mmd_experiment(
        cfg,
        mmds,
        graph_type,
        graph_extraction_param,
        "taper",
        descriptor,
        kernel_params=None,
    )


def gaussian_noise_perturbation_linear_kernel(
    cfg,
    perturbed,
    unperturbed,
    experiment_steps,
    graph_type,
    graph_extraction_param,
    descriptor,
    **kwargs,
):
    log.info("Perturbing proteins with gaussian noise.")

    def gaussian_noise_perturbation_worker(
        perturb, perturbed, unperturbed, graph_type, descriptor
    ):
        log.info(f"Pertubation set to {perturb}.")
        perturbation = (
            f"gaussiannoise_{perturb}",
            GaussianNoise(
                noise_mean=0,
                noise_std=perturb,
                random_state=hash(
                    str(perturbed)
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = point_cloud_perturbation_worker(
            cfg,
            experiment_steps,
            perturbation,
            unperturbed,
            perturbed,
            graph_type,
            descriptor,
        )
        mmd_pack = {"perturb": perturb, "mmd": mmd_runs}

        return mmd_pack

    mmds = distribute_function(
        func=gaussian_noise_perturbation_worker,
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
        graph_type=graph_type,
        descriptor=descriptor,
    )

    save_mmd_experiment(
        cfg,
        mmds,
        graph_type,
        graph_extraction_param,
        "gaussian_noise",
        descriptor,
        kernel_params=None,
    )


def linear_kernel_experiment_graph_perturbation(
    cfg: DictConfig,
    graph_type: str,
    graph_extraction_param: int,
    descriptor: str,
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

    if graph_type == "knn_graph":
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

    if descriptor == "degree_histogram":
        base_feature_steps.append(
            (
                "degree_histogram",
                DegreeHistogram(
                    graph_type=graph_type,
                    n_bins=100,
                    bin_range=(1, 100),
                    n_jobs=cfg.compute.n_jobs,
                    verbose=cfg.debug.verbose,
                ),
            )
        )
    elif descriptor == "clustering_histogram":
        base_feature_steps.append(
            (
                "clustering_histogram",
                ClusteringHistogram(
                    graph_type=graph_type,
                    n_bins=100,
                    n_jobs=cfg.compute.n_jobs,
                    verbose=cfg.debug.verbose,
                ),
            )
        )
    elif descriptor == "laplacian_spectrum_histogram":
        base_feature_steps.append(
            (
                "laplacian_spectrum_histogram",
                LaplacianSpectrum(
                    graph_type=graph_type,
                    n_bins=100,
                    n_jobs=cfg.compute.n_jobs,
                    verbose=cfg.debug.verbose,
                    bin_range=(0, 100),
                ),
            )
        )

    unperturbed = load_proteins_from_config(cfg, perturbed=False)
    unperturbed = pipeline.Pipeline(base_feature_steps).fit_transform(
        unperturbed
    )
    perturbed = load_proteins_from_config(cfg, perturbed=True)
    perturbed = pipeline.Pipeline([base_feature_steps[0]]).fit_transform(
        perturbed
    )

    log.info("Compute remove_edges")
    twist_perturbation_linear_kernel(
        cfg,
        perturbed,
        unperturbed,
        base_feature_steps,
        graph_type,
        graph_extraction_param,
        descriptor=descriptor,
    )

    shear_perturbation_linear_kernel(
        cfg,
        perturbed,
        unperturbed,
        base_feature_steps,
        graph_type,
        graph_extraction_param,
        descriptor=descriptor,
    )

    taper_perturbation_linear_kernel(
        cfg,
        perturbed,
        unperturbed,
        base_feature_steps,
        graph_type,
        graph_extraction_param,
        descriptor=descriptor,
    )

    gaussian_noise_perturbation_linear_kernel(
        cfg,
        perturbed,
        unperturbed,
        base_feature_steps,
        graph_type,
        graph_extraction_param,
        descriptor=descriptor,
    )


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
    log.info(DATA_HOME)
    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Start with Weisfeiler-Lehman-based-experiments.
    # outside for loops for n_iters and k.

    descriptors = [
        "degree_histogram",
        "clustering_histogram",
        "laplacian_spectrum_histogram",
    ]
    for descriptor in descriptors:
        for k in cfg.meta.representations[1]["knn_graph"]:
            linear_kernel_experiment_graph_perturbation(
                cfg=cfg,
                graph_type="knn_graph",
                graph_extraction_param=k,
                descriptor=descriptor,
            )
        for eps in cfg.meta.representations[1]["eps_graph"]:
            linear_kernel_experiment_graph_perturbation(
                cfg=cfg,
                graph_type="eps_graph",
                graph_extraction_param=eps,
                descriptor=descriptor,
            )


if __name__ == "__main__":
    main()
