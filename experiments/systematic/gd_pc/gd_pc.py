#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""gd_pc.py

"""

import logging
import os
import random
from enum import unique
from pathlib import Path
from re import A
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
from proteinmetrics.kernels import GaussianKernel, LinearKernel
from proteinmetrics.loaders import list_pdb_files, load_descriptor, load_graphs
from proteinmetrics.paths import DATA_HOME, ECOLI_PROTEOME, HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates, Sequence
from proteinmetrics.perturbations import (
    GaussianNoise,
    Mutation,
    Shear,
    Taper,
    Twist,
)
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

    perturbed_protein_names = idx2name2run(cfg, perturbed=True)
    unperturbed_protein_names = idx2name2run(cfg, perturbed=False)

    # For every run - compute x-y
    pre_computed_products = list()
    for run in range(cfg.meta.n_runs):
        log.info(f"Run {run}")
        unperturbed_run = filter_protein_using_name(
            unperturbed, unperturbed_protein_names[run].tolist()
        )
        perturbed_run = filter_protein_using_name(
            perturbed, perturbed_protein_names[run].tolist()
        )

        if descriptor == "distance_histogram":
            unperturbed_descriptor_run = np.asarray(
                [protein.distance_hist for protein in unperturbed_run]
            )
            perturbed_descriptor_run = np.asarray(
                [protein.distance_hist for protein in perturbed_run]
            )

        elif descriptor == "dihedral_angles_histogram":
            unperturbed_descriptor_run = np.asarray(
                [protein.phi_psi_angles for protein in unperturbed_run]
            )
            perturbed_descriptor_run = np.asarray(
                [protein.phi_psi_angles for protein in perturbed_run]
            )
        else:
            unperturbed_descriptor_run = load_descriptor(
                unperturbed_run, graph_type=graph_type, descriptor=descriptor
            )
            perturbed_descriptor_run = load_descriptor(
                perturbed_run, graph_type=graph_type, descriptor=descriptor
            )

        products = {
            # The np.ones is used here because
            # exp(sigma*(x-x)**2) = 1(n x n)
            "K_XX": pairwise_distances(
                unperturbed_descriptor_run, unperturbed_descriptor_run,
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
        for run in range(cfg.meta.n_runs):
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
    for run in range(cfg.meta.n_runs):
        log.info(f"Run {run}")
        unperturbed_run = filter_protein_using_name(
            unperturbed, unperturbed_protein_names[run].tolist()
        )
        perturbed_run = filter_protein_using_name(
            perturbed, perturbed_protein_names[run].tolist()
        )

        if descriptor == "distance_histogram":
            unperturbed_descriptor_run = np.asarray(
                [protein.distance_hist for protein in unperturbed_run]
            )
            perturbed_descriptor_run = np.asarray(
                [protein.distance_hist for protein in perturbed_run]
            )

        elif descriptor == "dihedral_angles_histogram":
            unperturbed_descriptor_run = np.asarray(
                [protein.phi_psi_angles for protein in unperturbed_run]
            )
            perturbed_descriptor_run = np.asarray(
                [protein.phi_psi_angles for protein in perturbed_run]
            )
        else:
            unperturbed_descriptor_run = load_descriptor(
                unperturbed_run,
                graph_type=graph_type,
                descriptor=descriptor,
            )
            perturbed_descriptor_run = load_descriptor(
                perturbed_run, graph_type=graph_type, descriptor=descriptor
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


def save_mmd_experiment(
    cfg,
    mmds,
    graph_type,
    graph_extraction_param,
    perturbation_type,
    descriptor: str,
):
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
        / cfg.paths.fixed_length_kernels
        / graph_type
        / str(graph_extraction_param)
        / perturbation_type
        / descriptor
    )
    make_dir(target_dir)
    mmds.to_csv(target_dir / f"{perturbation_type}_mmds.csv")

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

    def twist_perturbation_worker(
        perturb, perturbed, unperturbed, graph_type, descriptor
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
        mmd_df = pd.DataFrame(mmd_runs).assign(perturb=perturb)

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
        mmd_df = pd.DataFrame(mmd_runs).assign(perturb=perturb)
        return mmd_df

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
        mmd_df = pd.DataFrame(mmd_runs).assign(perturb=perturb)
        return mmd_df

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
        mmd_df = pd.DataFrame(mmd_runs).assign(perturb=perturb)
        return mmd_df

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
    )


def mutation_perturbation_linear_kernel(
    cfg,
    perturbed,
    unperturbed,
    experiment_steps,
    graph_type,
    graph_extraction_param,
    descriptor,
    **kwargs,
):
    log.info("Perturbing proteins with mutation.")

    def mutation_perturbation_worker(
        perturb, perturbed, unperturbed, graph_type, descriptor
    ):
        log.info(f"Pertubation set to {perturb}.")
        perturbation = (
            f"mutation_{perturb}",
            Mutation(
                p_mutate=perturb,
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
        mmd_df = pd.DataFrame(mmd_runs).assign(perturb=perturb)
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
        graph_type=graph_type,
        descriptor=descriptor,
    )

    save_mmd_experiment(
        cfg,
        mmds,
        graph_type,
        graph_extraction_param,
        "mutation",
        descriptor,
    )


def fixed_length_kernel_experiment_graph_perturbation(
    cfg: DictConfig,
    graph_type: str,
    graph_extraction_param: int,
    descriptor: str,
    perturbation: str,
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
    elif graph_type == "pc_descriptor":
        pass

    else:
        raise ValueError(f"Unknown graph type {graph_type}")

    if descriptor == "degree_histogram":
        base_feature_steps.append(
            (
                "degree_histogram",
                DegreeHistogram(
                    graph_type=graph_type,
                    n_bins=cfg.descriptors.degree_histogram.n_bins,
                    bin_range=(
                        cfg.descriptors.degree_histogram.bin_range.min,
                        cfg.descriptors.degree_histogram.bin_range.max,
                    ),
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
                    n_bins=cfg.descriptors.laplacian_spectrum.n_bins,
                    n_jobs=cfg.compute.n_jobs,
                    verbose=cfg.debug.verbose,
                    bin_range=(
                        cfg.descriptors.laplacian_spectrum.bin_range.min,
                        cfg.descriptors.laplacian_spectrum.bin_range.max,
                    ),
                ),
            )
        )
    elif descriptor == "distance_histogram":
        base_feature_steps.append(
            (
                "distance_histogram",
                DistanceHistogram(
                    n_bins=cfg.descriptors.distance_histogram.n_bins,
                    n_jobs=cfg.compute.n_jobs,
                    verbose=cfg.debug.verbose,
                    bin_range=(
                        cfg.descriptors.distance_histogram.bin_range.min,
                        cfg.descriptors.distance_histogram.bin_range.max,
                    ),
                ),
            )
        )
    elif descriptor == "dihedral_angles_histogram":
        base_feature_steps[0] = (
            (
                "coordinates",
                Coordinates(
                    granularity="backbone",
                    n_jobs=cfg.compute.n_jobs,
                    verbose=True,
                ),
            ),
        )[0]

        base_feature_steps.append(
            (
                "dihedral_angles",
                RamachandranAngles(
                    from_pdb=False,
                    n_bins=cfg.descriptors.dihedral_anlges.n_bins,
                    n_jobs=cfg.compute.n_jobs,
                    verbose=cfg.debug.verbose,
                ),
            )
        )
        pass
    else:
        raise ValueError("Unknown descriptor")

    unperturbed = load_proteins_from_config(cfg, perturbed=False)
    unperturbed = pipeline.Pipeline(base_feature_steps).fit_transform(
        unperturbed
    )
    perturbed = load_proteins_from_config(cfg, perturbed=True)
    perturbed = pipeline.Pipeline([base_feature_steps[0]]).fit_transform(
        perturbed
    )

    if perturbation == "twist":
        twist_perturbation_linear_kernel(
            cfg,
            perturbed,
            unperturbed,
            base_feature_steps,
            graph_type,
            graph_extraction_param,
            descriptor=descriptor,
        )
    elif perturbation == "shear":
        shear_perturbation_linear_kernel(
            cfg,
            perturbed,
            unperturbed,
            base_feature_steps,
            graph_type,
            graph_extraction_param,
            descriptor=descriptor,
        )

    elif perturbation == "taper":
        taper_perturbation_linear_kernel(
            cfg,
            perturbed,
            unperturbed,
            base_feature_steps,
            graph_type,
            graph_extraction_param,
            descriptor=descriptor,
        )

    elif perturbation == "gaussian_noise":
        gaussian_noise_perturbation_linear_kernel(
            cfg,
            perturbed,
            unperturbed,
            base_feature_steps,
            graph_type,
            graph_extraction_param,
            descriptor=descriptor,
        )
    elif perturbation == "mutation":
        mutation_perturbation_linear_kernel(
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

    fixed_length_kernel_experiment_graph_perturbation(
        cfg=cfg,
        graph_type=cfg.graph_type,
        graph_extraction_param=cfg.graph_extraction_parameter,
        descriptor=cfg.descriptor,
        perturbation=cfg.perturbation,
    )


if __name__ == "__main__":
    main()
