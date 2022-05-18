#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""time_estimator.py

Here we want to get an accurate time estimate of the computation time for some tasks.

"""

import time

import hydra
import pandas as pd
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf.dictconfig import DictConfig
from pyprojroot import here

from proteinggnnmetrics.descriptors import (
    ESM,
    ClusteringHistogram,
    DegreeHistogram,
    LaplacianSpectrum,
    TopologicalDescriptor,
)
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.kernels import (
    GaussianKernel,
    LinearKernel,
    PersistenceFisherKernel,
)
from proteinggnnmetrics.loaders import (
    list_pdb_files,
    load_descriptor,
    load_graphs,
)
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.functions import (
    get_longest_protein_dummy_sequence,
)

N_JOBS = 4


@timeit
def tda_benchmark(pdb_files):
    base_feature_steps = [
        ("coordinates", Coordinates(granularity="CA", n_jobs=N_JOBS),),
        ("contact_map", ContactMap(n_jobs=N_JOBS, verbose=True,),),
        (
            "epsilon_graph",
            EpsilonGraph(n_jobs=N_JOBS, epsilon=8, verbose=True,),
        ),
        (
            "tda",
            TopologicalDescriptor(
                "diagram",
                epsilon=0.01,
                n_bins=100,
                order=2,
                n_jobs=N_JOBS,
                landscape_layers=1,
                verbose=True,
            ),
        ),
    ]
    start = time.perf_counter()
    proteins = pipeline.Pipeline(
        base_feature_steps, verbose=True
    ).fit_transform(pdb_files)
    time_elapsed = time.perf_counter() - start
    return proteins, time_elapsed


@timeit
def esm_benchmark(pdb_files):
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=N_JOBS, verbose=True,),
        ),
        (
            "esm",
            ESM(
                size="M",
                n_jobs=N_JOBS,
                verbose=True,
                longest_sequence=get_longest_protein_dummy_sequence(
                    pdb_files, N_JOBS
                ),
                n_chunks=2,
            ),
        ),
    ]
    start = time.perf_counter()
    proteins = pipeline.Pipeline(
        base_feature_steps, verbose=True
    ).fit_transform(pdb_files)
    time_elapsed = time.perf_counter() - start
    return proteins, time_elapsed


@timeit
def graphs_benchmark(pdb_files):
    base_feature_steps = [
        ("coordinates", Coordinates(granularity="CA", n_jobs=N_JOBS),),
        ("contact map", ContactMap(metric="euclidean", n_jobs=N_JOBS,),),
        ("epsilon graph", EpsilonGraph(epsilon=8, n_jobs=N_JOBS),),
    ]
    start = time.perf_counter()
    proteins = pipeline.Pipeline(
        base_feature_steps, verbose=True
    ).fit_transform(pdb_files)
    time_elapsed = time.perf_counter() - start
    return proteins, time_elapsed


def reps_benchmarks(pdb_files):
    tda_proteins, tda_time_elapsed = tda_benchmark(pdb_files)
    esm_proteins, esm_time_elapsed = esm_benchmark(pdb_files)
    graph_proteins, graph_time_elapsed = graphs_benchmark(pdb_files)
    proteins = list()
    for tda_protein, esm_protein, graph_protein in zip(
        tda_proteins, esm_proteins, graph_proteins
    ):
        protein = esm_protein
        protein.descriptors["contact_graph"][
            "diagram"
        ] = tda_protein.descriptors["contact_graph"]["diagram"]
        protein.graphs["eps_graph"] = graph_protein.graphs["eps_graph"]
        proteins.append(protein)

    return proteins, tda_time_elapsed, esm_time_elapsed, graph_time_elapsed


@timeit
def degree_histogram_benchmark(proteins):
    start = time.perf_counter()
    proteins = DegreeHistogram(
        graph_type="eps_graph", n_bins=100, n_jobs=N_JOBS, verbose=True,
    ).fit_transform(proteins)
    time_elapsed = time.perf_counter() - start
    return proteins, time_elapsed


@timeit
def clustering_histogram_benchmark(proteins):
    start = time.perf_counter()
    proteins = ClusteringHistogram(
        graph_type="eps_graph",
        n_bins=100,
        density=True,
        n_jobs=N_JOBS,
        verbose=True,
    ).fit_transform(proteins)
    time_elapsed = time.perf_counter() - start
    return proteins, time_elapsed


@timeit
def laplacian_spectrum_histogram_benchmark(proteins):
    start = time.perf_counter()
    proteins = LaplacianSpectrum(
        graph_type="eps_graph", n_bins=100, n_jobs=N_JOBS, verbose=True,
    ).fit_transform(proteins)
    time_elapsed = time.perf_counter() - start
    return proteins, time_elapsed


def desc_benchmarks(proteins):
    degree_proteins, degree_time_elapsed = degree_histogram_benchmark(proteins)
    (
        clustering_proteins,
        clustering_time_elapsed,
    ) = clustering_histogram_benchmark(proteins)
    (
        laplacian_proteins,
        laplacian_time_elapsed,
    ) = laplacian_spectrum_histogram_benchmark(proteins)
    proteins = list()
    for degree_protein, clustering_protein, laplacian_protein in zip(
        degree_proteins, clustering_proteins, laplacian_proteins
    ):
        protein = degree_protein
        protein.descriptors["eps_graph"][
            "clustering"
        ] = clustering_protein.descriptors["clustering"]
        protein.descriptors["eps_graph"][
            "laplacian"
        ] = laplacian_protein.descriptors["eps_graph"]["laplacian"]
        proteins.append(protein)

    return (
        proteins,
        degree_time_elapsed,
        clustering_time_elapsed,
        laplacian_time_elapsed,
    )


@timeit
def linear_k_benchmark(left, right):
    start = time.perf_counter()
    kernel = LinearKernel().compute_matrix(left, right)
    time_elapsed = time.perf_counter() - start
    return kernel, time_elapsed


@timeit
def gaussian_k_benchmark(left, right):
    # this this the worst-case scenario
    start = time.perf_counter()
    kernel = GaussianKernel(
        sigma=1, pre_computed_difference=False, return_product=False
    ).compute_matrix(left, right)
    time_elapsed = time.perf_counter() - start
    return kernel, time_elapsed


@timeit
def wl_k_benchmark(left, right):
    start = time.perf_counter()
    kernel = WeisfeilerLehmanKernel(n_jobs=10, n_iter=4).compute_matrix(
        left, right
    )
    time_elapsed = time.perf_counter() - start
    return kernel, time_elapsed


@timeit
def persistence_fisher_k_benchmark(left, right):
    start = time.perf_counter()
    kernel = PersistenceFisherKernel(n_jobs=10, n_iter=4).compute_matrix(
        left, right
    )
    time_elapsed = time.perf_counter() - start
    return kernel, time_elapsed


def ker_benchmarks(proteins):

    half = int(len(proteins) / 2)
    left = proteins[:half]
    right = proteins[half:]

    # Get fixed-length vector reps
    left_deg_hist = load_descriptor(left, "degree_histogram", "eps_graph")
    right_deg_hist = load_descriptor(right, "degree_histogram", "eps_graph")

    # Linear
    _, linear_k_time_elapsed = linear_k_benchmark(
        left_deg_hist, right_deg_hist
    )

    # Gaussian
    _, gaussian_k_time_elapsed = gaussian_k_benchmark(
        left_deg_hist, right_deg_hist
    )

    # W-L
    left_graphs = load_graphs(proteins[:half], "eps_graph")
    right_graphs = load_graphs(proteins[half:], "eps_graph")
    _, wl_k_time_elapsed = wl_k_benchmark(left_graphs, right_graphs)

    # persistence_fisher
    left_pd = [
        protein.descriptors["contact_graph"]["persistence_diagram"]
        for protein in proteins[:half]
    ]
    right_pd = [
        protein.descriptors["contact_graph"]["persistence_diagram"]
        for protein in proteins[half:]
    ]
    _, pf_k_time_elapsed = persistence_fisher_k_benchmark(left_pd, right_pd)
    return (
        linear_k_time_elapsed,
        gaussian_k_time_elapsed,
        wl_k_time_elapsed,
        pf_k_time_elapsed,
    )


@hydra.main(config_path=str(here()) + "/conf", config_name="conf")
def main(cfg: DictConfig):
    # Potential performance improvements in the real world:
    # - not a representative RAM footprint
    # - not a fully representative scaling/sample size

    # Reprensentations benchmarks
    (
        proteins,
        tda_time_elapsed,
        esm_time_elapsed,
        graph_time_elapsed,
    ) = reps_benchmarks(list_pdb_files(HUMAN_PROTEOME)[:5])

    representation_benchmarks = pd.DataFrame(
        data=[tda_time_elapsed, esm_time_elapsed, graph_time_elapsed],
        index=["tda", "esm", "graph"],
        columns=["representation"],
    )

    representation_benchmarks.to_csv(
        here()
        / cfg.meta.data.time_estimates_dir
        / "representation_benchmarks.csv"
    )

    # Descriptor benchmarks

    (
        proteins,
        degree_time_elapsed,
        clustering_time_elapsed,
        laplacian_time_elapsed,
    ) = desc_benchmarks(proteins)

    descriptor_benchmarks = pd.DataFrame(
        data=[
            degree_time_elapsed,
            clustering_time_elapsed,
            laplacian_time_elapsed,
        ],
        index=["degree", "clustering", "laplacian"],
        columns=["descriptor"],
    )
    representation_benchmarks.to_csv(
        here() / cfg.meta.data.time_estimates_dir / "descriptor_benchmarks.csv"
    )

    # kernel benchmarks
    (
        linear_k_time_elapsed,
        gaussian_k_time_elapsed,
        wl_k_time_elapsed,
        pf_k_time_elapsed,
    ) = ker_benchmarks(proteins)

    kernel_benchmarks = pd.DataFrame(
        data=[
            linear_k_time_elapsed,
            gaussian_k_time_elapsed,
            wl_k_time_elapsed,
            pf_k_time_elapsed,
        ],
        index=["linear", "gaussian", "wl", "pf"],
        columns=["kernel"],
    )
    kernel_benchmarks.to_csv(
        here() / cfg.meta.data.time_estimates_dir / "kernel_benchmarks.csv"
    )


if __name__ == "__main__":
    main()
