#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""kernel_matrix_computations.py

This is to test kernel matrix computations

"""

import os
from collections import Counter
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
from grakel.kernels.vertex_histogram import VertexHistogram
from numpy.testing import assert_array_less

from proteinggnnmetrics.kernels import LinearKernel, WeisfeilerLehmanKernel
from proteinggnnmetrics.loaders import (
    load_descriptor,
    load_graphs,
    load_proteins,
)
from proteinggnnmetrics.paths import CACHE_DIR
from proteinggnnmetrics.protein import Protein
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.functions import (
    configure,
    distribute_function,
    flatten_lists,
)

config = configure()

default_eigvalue_precision = float("-1e-5")


def positive_eig(K):
    """Assert true if the calculated kernel matrix is valid."""
    min_eig = np.real(np.min(np.linalg.eig(K)[0]))
    assert_array_less(default_eigvalue_precision, min_eig)


def get_hash(G):
    return dict(
        Counter(
            flatten_lists(
                list(
                    nx.weisfeiler_lehman_subgraph_hashes(
                        G, iterations=10,
                    ).values()
                )
            )
        )
    )


def compute_wl_hashes(protein):
    protein.set_weisfeiler_lehman_hashes(graph_type="knn_graph", n_iter=3)
    return protein


def generate_data(seed, n_graphs):
    """Generates data for testing"""

    # Generate n_graphs graphs

    X = list()
    Y = list()
    for n in range(n_graphs):
        X.append(get_hash(nx.erdos_renyi_graph(n=4, p=0.8, seed=seed)))
        Y.append(get_hash(nx.erdos_renyi_graph(n=4, p=0.5, seed=seed)))
    return X, Y


def generate_simple_data():
    s_1 = {"a": 2, "b": 3, "c": 1, "d": 10}
    s_2 = {"a": 2, "b": 1, "e": 10, "f": 20}
    s_3 = {"a": 2, "b": 1, "c": 2, "f": 1}
    s_4 = {"b": 1, "c": 2, "f": 1, "g": 10}
    s_5 = {"g": 10, "h": 20, "i": 1, "j": 2}
    return [s_1, s_2, s_3, s_4, s_5]


@timeit
@measure_memory
def precomputed_vectorized_biased(X, Y):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=config["COMPUTE"]["N_JOBS"],
        n_iter=3,
        biased=True,
        pre_computed_hash=True,
        vectorized=True,
    )
    KXY = wl_kernel.fit_transform(X, Y)
    positive_eig(KXY)
    print(KXY)


@timeit
@measure_memory
def precomputed_custom_biased(X, Y):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=config["COMPUTE"]["N_JOBS"],
        n_iter=3,
        biased=True,
        pre_computed_hash=True,
        vectorized=False,
    )
    KXY = wl_kernel.fit_transform(X, Y)
    positive_eig(KXY)
    print(KXY)


@timeit
@measure_memory
def precomputed_naive_biased(X, Y):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=int(config["COMPUTE"]["N_JOBS"]),
        n_iter=3,
        biased=True,
        pre_computed_hash=False,
        vectorized=False,
        normalize=False,
    )
    KXY = wl_kernel.fit_transform(X, Y)
    positive_eig(KXY)
    print(KXY)


def main():
    proteins = load_proteins(CACHE_DIR / "sample_human_proteome_alpha_fold")
    proteins = distribute_function(
        compute_wl_hashes,
        proteins,
        int(config["COMPUTE"]["N_JOBS"]),
        "Computing Weisfeiler-Lehman Hashes",
        show_tqdm=False,
    )
    hashes = [
        protein.descriptors["knn_graph"]["weisfeiler-lehman-hist"]
        for protein in proteins
    ]

    precomputed_custom_biased(hashes, hashes)
    # precomputed_vectorized_biased(hashes, hashes)

    graphs = [protein.graphs["knn_graph"] for protein in proteins]
    precomputed_naive_biased(graphs, graphs)


if __name__ == "__main__":
    main()
