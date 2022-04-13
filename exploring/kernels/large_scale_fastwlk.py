#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""large_scale_fastwlk.py

Large scale fastwlk test.

"""
from datetime import datetime
from tkinter import N

import numpy as np
import plotly.graph_objects as go
from fastwlk.kernel import WeisfeilerLehmanKernel
from grakel.graph_kernels import WeisfeilerLehman
from gtda import pipeline

from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.loaders import list_pdb_files, load_graphs
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.functions import networkx2grakel, positive_eig

N_JOBS = 10


@timeit
@measure_memory
def fastwlk_test(graphs):
    K = WeisfeilerLehmanKernel(
        n_jobs=N_JOBS,
        n_iter=5,
        normalize=False,
        biased=True,
        verbose=False,
    ).compute_gram_matrix(graphs)


@timeit
@measure_memory
def grakel_test(graphs):
    graphs = networkx2grakel(graphs)
    K = WeisfeilerLehman(n_jobs=N_JOBS, n_iter=5).fit(graphs).transform(graphs)


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=N_JOBS, verbose=False),
        ),
        (
            "contact map",
            ContactMap(metric="euclidean", n_jobs=N_JOBS, verbose=False),
        ),
        (
            "epsilon graph",
            EpsilonGraph(epsilon=20, n_jobs=N_JOBS, verbose=False),
        ),
    ]

    base_feature_pipeline = pipeline.Pipeline(
        base_feature_steps, verbose=False
    )
    proteins = base_feature_pipeline.fit_transform(pdb_files[:20])
    proteins_1 = base_feature_pipeline.fit_transform(pdb_files[20:25])
    graphs = load_graphs(proteins, graph_type="eps_graph")
    graphs_1 = load_graphs(proteins_1, graph_type="eps_graph")
    # grakel_test(graphs)
    # fastwlk_test(graphs)
    K_XX = WeisfeilerLehmanKernel(
        n_jobs=N_JOBS,
        n_iter=5,
        normalize=False,
        biased=True,
        verbose=False,
    ).compute_gram_matrix(graphs, graphs)
    K_YY = WeisfeilerLehmanKernel(
        n_jobs=N_JOBS,
        n_iter=5,
        normalize=False,
        biased=True,
        verbose=False,
    ).compute_gram_matrix(graphs_1, graphs_1)
    K_XY = WeisfeilerLehmanKernel(
        n_jobs=N_JOBS,
        n_iter=5,
        normalize=False,
        biased=True,
        verbose=False,
    ).compute_gram_matrix(graphs, graphs_1)
    full_K = np.zeros(
        (K_XX.shape[0] + K_YY.shape[0], K_XX.shape[0] + K_YY.shape[0])
    )
    full_K[: K_XX.shape[0], : K_XX.shape[0]] = K_XX
    full_K[K_XX.shape[0] :, K_XX.shape[0] :] = K_YY
    full_K[: K_XX.shape[0], K_XX.shape[0] :] = K_XY
    positive_eig(full_K)


if __name__ == "__main__":
    main()
