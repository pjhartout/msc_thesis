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
from grakel import VertexHistogram
from numpy.testing import assert_array_less
from sklearn.metrics.pairwise import linear_kernel

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
    assert_array_less(min_eig, default_eigvalue_precision)


def compute_wl_hashes(protein):
    protein.set_weisfeiler_lehman_hashes(graph_type="knn_graph", n_iter=3)
    return protein


def generate_simple_data():
    s_1 = {"a": 2, "b": 3, "c": 1, "d": 10}
    s_2 = {"a": 2, "b": 1, "e": 10, "f": 20}
    s_3 = {"a": 2, "b": 1, "c": 2, "f": 1}
    s_4 = {"b": 1, "c": 2, "f": 1, "g": 10}
    s_5 = {"g": 10, "h": 20, "i": 1, "j": 2}
    return [s_1, s_2, s_3, s_4, s_5]


@timeit
@measure_memory
def precomputed_custom_biased(X, Y):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=config["COMPUTE"]["N_JOBS"],
        n_iter=4,
        biased=True,
        pre_computed_hash=True,
        vectorized=False,
    )
    KXY = wl_kernel.fit_transform(X, Y)
    # positive_eig(KXY)
    # Print shape of KXY
    print(f"Custom: {KXY.shape}")
    return KXY


@timeit
@measure_memory
def precomputed_naive_biased(X, Y):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=None,
        n_iter=3,
        biased=True,
        base_graph_kernel=VertexHistogram,
        pre_computed_hash=False,
        vectorized=False,
        normalize=False,
    )
    KXY = wl_kernel.fit_transform(X, Y)
    # positive_eig(KXY)
    # Print shape of KXY
    print(f"Custom: {KXY.shape}")
    return KXY


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

    precomp = precomputed_custom_biased(hashes[:30], hashes[30:])

    graphs = [protein.graphs["knn_graph"] for protein in proteins]
    naive = precomputed_naive_biased(graphs[:30], graphs[30:])
    print("Precomputed:", precomp.shape)
    print("Naive:", naive.shape)


if __name__ == "__main__":
    main()
