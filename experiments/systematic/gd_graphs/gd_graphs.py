#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""gd_graphs.py

"""

import logging
import os
import random
from pathlib import Path
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
from proteinmetrics.loaders import list_pdb_files, load_graphs
from proteinmetrics.paths import DATA_HOME, ECOLI_PROTEOME, HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates, Sequence
from proteinmetrics.perturbations import AddEdges, RemoveEdges, RewireEdges
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


def graph_perturbation_worker(
    cfg,
    experiment_steps,
    perturbation,
    unperturbed,
    perturbed,
    graph_type,
):
    experiment_steps_perturbed = experiment_steps[1:]
    experiment_steps_perturbed.append(perturbation)
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

        unperturbed_graphs = load_graphs(
            unperturbed_run, graph_type=graph_type
        )
        perturbed_graphs = load_graphs(perturbed_run, graph_type=graph_type)

        if cfg.debug.reduce_data:
            unperturbed_graphs = unperturbed_graphs[
                : cfg.debug.sample_data_size
            ]
            perturbed_graphs = perturbed_graphs[: cfg.debug.sample_data_size]

        log.info("Computing the kernel.")

        mmd = MaximumMeanDiscrepancy(
            biased=False,
            squared=True,
            kernel=LinearKernel(
                n_jobs=cfg.compute.n_jobs,
                normalize=True,
            ),  # type: ignore
        ).compute(unperturbed_graphs, perturbed_graphs)
        mmd_runs.append(mmd)
    return mmd_runs


def save_mmd_experiment(
    cfg,
    mmds,
    graph_type,
    graph_extraction_param,
    perturbation_type,
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


def remove_edge_perturbation_linear_kernel(
    cfg,
    perturbed,
    unperturbed,
    experiment_steps,
    graph_type,
    graph_extraction_param,
    **kwargs,
):
    log.info("Perturbing proteins with RemoveEdges.")

    def remove_edges_perturbation_worker(
        p_perturb, perturbed, unperturbed, graph_type
    ):
        log.info(f"Pertubation set to {p_perturb}.")
        perturbation = (
            f"removedge_{p_perturb}",
            RemoveEdges(
                p_remove=p_perturb,
                graph_type=graph_type,
                random_state=np.random.RandomState(
                    divmod(hash(str(perturbed)), 42)[1]
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = graph_perturbation_worker(
            cfg,
            experiment_steps,
            perturbation,
            unperturbed,
            perturbed,
            graph_type,
        )
        mmd_pack = {"perturb": p_perturb, "mmd": mmd_runs}

        return mmd_pack

    mmds = distribute_function(
        func=remove_edges_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.remove_edges.min,
            cfg.perturbations.remove_edges.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_perturb,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="Remove edges experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
        graph_type=graph_type,
    )

    save_mmd_experiment(
        cfg,
        mmds,
        graph_type,
        graph_extraction_param,
        "removedge",
    )


def add_edge_perturbation_linear_kernel(
    cfg,
    perturbed,
    unperturbed,
    experiment_steps,
    graph_type,
    graph_extraction_param,
    **kwargs,
):
    log.info("Perturbing proteins with AddEdges.")

    def add_edges_perturbation_worker(
        p_perturb, perturbed, unperturbed, graph_type
    ):
        log.info(f"Pertubation set to {p_perturb}.")
        perturbation = (
            f"addedge_{p_perturb}",
            AddEdges(
                p_add=p_perturb,
                graph_type=graph_type,
                random_state=np.random.RandomState(
                    divmod(hash(str(perturbed)), 42)[1]
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = graph_perturbation_worker(
            cfg,
            experiment_steps,
            perturbation,
            unperturbed,
            perturbed,
            graph_type,
        )
        mmd_pack = {"perturb": p_perturb, "mmd": mmd_runs}

        return mmd_pack

    mmds = distribute_function(
        func=add_edges_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.add_edges.min,
            cfg.perturbations.add_edges.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_perturb,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="Add edges experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
        graph_type=graph_type,
    )

    save_mmd_experiment(
        cfg,
        mmds,
        graph_type,
        graph_extraction_param,
        "addedge",
    )


def rewire_edge_perturbation_linear_kernel(
    cfg,
    perturbed,
    unperturbed,
    experiment_steps,
    graph_type,
    graph_extraction_param,
    **kwargs,
):
    log.info("Perturbing proteins with RewireEdges.")

    def rewire_edges_perturbation_worker(
        p_perturb, perturbed, unperturbed, graph_type
    ):
        log.info(f"Pertubation set to {p_perturb}.")
        perturbation = (
            f"rewireedge_{p_perturb}",
            RewireEdges(
                p_rewire=p_perturb,
                graph_type=graph_type,
                random_state=np.random.RandomState(
                    divmod(hash(str(perturbed)), 42)[1]
                ),  # The seed is the same as long as the paths is the same.
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        )
        mmd_runs = graph_perturbation_worker(
            cfg,
            experiment_steps,
            perturbation,
            unperturbed,
            perturbed,
            graph_type,
        )
        mmd_pack = {"perturb": p_perturb, "mmd": mmd_runs}

        return mmd_pack

    mmds = distribute_function(
        func=rewire_edges_perturbation_worker,
        X=np.linspace(
            cfg.perturbations.rewire_edges.min,
            cfg.perturbations.rewire_edges.max,
            cfg.perturbations.n_perturbations,
        ),
        n_jobs=cfg.compute.n_parallel_perturb,
        show_tqdm=cfg.debug.verbose,
        tqdm_label="Rewire edges experiment",
        perturbed=perturbed,
        unperturbed=unperturbed,
        graph_type=graph_type,
    )

    save_mmd_experiment(
        cfg,
        mmds,
        graph_type,
        graph_extraction_param,
        "rewireedge",
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
    unperturbed = load_proteins_from_config(cfg, perturbed=False)
    unperturbed = pipeline.Pipeline(base_feature_steps).fit_transform(
        unperturbed
    )
    perturbed = load_proteins_from_config(cfg, perturbed=True)
    perturbed = pipeline.Pipeline([base_feature_steps[0]]).fit_transform(
        perturbed
    )

    log.info("Compute remove_edges")
    remove_edge_perturbation_linear_kernel(
        cfg,
        perturbed,
        unperturbed,
        base_feature_steps,
        graph_type,
        graph_extraction_param,
    )

    add_edge_perturbation_linear_kernel(
        cfg,
        perturbed,
        unperturbed,
        base_feature_steps,
        graph_type,
        graph_extraction_param,
    )

    rewire_edge_perturbation_linear_kernel(
        cfg,
        perturbed,
        unperturbed,
        base_feature_steps,
        graph_type,
        graph_extraction_param,
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

    linear_kernel_experiment_graph_perturbation(
        cfg=cfg,
        graph_type=cfg.graph_type,
        graph_extraction_param=cfg.graph_extraction_param,
        descriptor=cfg.descriptor,
    )


if __name__ == "__main__":
    main()
