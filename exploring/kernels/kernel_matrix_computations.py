#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""kernel_matrix_computations.py

This is to test kernel matrix computations

"""

import itertools
import os
from collections import Counter
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
from gklearn.kernels.weisfeilerLehmanKernel import weisfeilerlehmankernel
from grakel import VertexHistogram, WeisfeilerLehman
from numpy.testing import assert_array_less
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
from typing_extensions import runtime

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
    distribute_function,
    flatten_lists,
    networkx2grakel,
)

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
        n_jobs=config["COMPUTE"]["N_JOBS"], n_iter=4, biased=True,
    )
    KXY = wl_kernel.fit_transform(X, Y)
    # positive_eig(KXY)
    # Print shape of KXY
    print(f"Custom: {KXY.shape}")
    return KXY


@timeit
@measure_memory
def grakel_test(X, Y):
    X = networkx2grakel(X)
    Y = networkx2grakel(Y)
    wl_kernel = WeisfeilerLehman(n_jobs=10, n_iter=3)
    KXY = wl_kernel.fit(X).transform(Y).T
    # positive_eig(KXY) Print shape of KXY
    print(f"Custom: {KXY.shape}")
    return KXY


def graphkit_test(X, Y):
    graphkit_matrix = np.zeros((len(X), len(Y)))
    for idx_1, idx_2 in tqdm(
        itertools.product(range(len(X)), range(len(Y))), total=len(X) * len(Y),
    ):
        # Compute W-L kernel between two graphs
        wl_kernel, runtime = weisfeilerlehmankernel(
            X[idx_1], Y[idx_2], node_label="residue", height=3, verbose=False,
        )
        graphkit_matrix[idx_1, idx_2] = wl_kernel[0, 1]
    return graphkit_matrix


def main():
    print("#### Loading data...")
    proteins = load_proteins(CACHE_DIR / "sample_human_proteome_alpha_fold")
    proteins = distribute_function(
        compute_wl_hashes,
        proteins,
        10,
        "Computing Weisfeiler-Lehman Hashes",
        show_tqdm=False,
    )
    hashes = [
        protein.descriptors["knn_graph"]["weisfeiler-lehman-hist"]
        for protein in proteins
    ]
    print("#### Self similarity test...")

    precomp = precomputed_custom_biased(hashes[:30], hashes[:30])
    graphs = [protein.graphs["knn_graph"] for protein in proteins]
    grakel_res = grakel_test(graphs[:30], graphs[:30])
    graphkit_res = graphkit_test(graphs[:30], graphs[:30])
    print(f"Precomputed: {precomp.shape}")
    print(f"Grakel: {grakel_res.shape}")
    print(f"Graphkit: {graphkit_res.shape}")
    print(
        f"Precomp-grakel sum difference: {np.sum(precomp) - np.sum(grakel_res)}"
    )
    print(
        f"Precomp-graphkit sum difference: {np.sum(precomp) - np.sum(graphkit_res)}"
    )

    print("#### Test differences between two different distributions")

    precomp = precomputed_custom_biased(hashes[:30], hashes[30:])
    graphs = [protein.graphs["knn_graph"] for protein in proteins]
    grakel_res = grakel_test(graphs[:30], graphs[30:])
    graphkit_res = graphkit_test(graphs[:30], graphs[30:])
    print(f"Precomputed: {precomp.shape}")
    print(f"Grakel: {grakel_res.shape}")
    print(f"Graphkit: {graphkit_res.shape}")
    print(
        f"Precomp-grakel sum difference: {np.sum(precomp) - np.sum(grakel_res)}"
    )
    print(
        f"Precomp-graphkit sum difference: {np.sum(precomp) - np.sum(graphkit_res)}"
    )


if __name__ == "__main__":
    main()
