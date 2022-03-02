#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""kernel_matrix_computations.py

This is to test kernel matrix computations

"""

import os
from pathlib import Path
from re import X
from typing import List

from proteinggnnmetrics.kernels import (
    LinearKernel,
    PreComputedWLKernel,
    WLKernel,
)
from proteinggnnmetrics.loaders import (
    load_descriptor,
    load_graphs,
    load_proteins,
)
from proteinggnnmetrics.paths import CACHE_DIR
from proteinggnnmetrics.protein import Protein
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.functions import configure, distribute_function

config = configure()


def compute_wl_hashes(protein):
    protein.set_weisfeiler_lehman_hashes(graph_type="knn_graph")
    return protein


@measure_memory
@timeit
def compute_naive_kernel(graphs):
    wl_kernel = WLKernel()
    KX = wl_kernel.fit_transform(graphs)
    return KX


@measure_memory
@timeit
def compute_precomp_kernel(proteins):
    proteins = distribute_function(
        compute_wl_hashes,
        proteins,
        "Computing Weisfeiler-Lehman Hashes",
        config["COMPUTE"]["N_JOBS"],
    )
    hashes = [
        protein.descriptors["knn_graph"]["weisfeiler-lehman-hist"]
        for protein in proteins
    ]

    precomp_wl = PreComputedWLKernel()
    KX = precomp_wl.fit_transform(hashes)
    return KX


def main():

    proteins = load_proteins(CACHE_DIR / "sample_human_proteome_alpha_fold")
    half_point = int(len(proteins) / 2)

    # 1. Generic kernel
    # 1.1 Linear kernel
    # First descriptor then computation
    # degree_histograms = load_descriptor(
    #     CACHE_DIR / "sample_human_proteome_alpha_fold",
    #     descriptor="degree_histogram",
    #     graph_type="knn_graph",
    # )
    # G_1 = degree_histograms[half_point:]
    # G_2 = degree_histograms[:half_point]
    # linear_kernel = LinearKernel(dense_output=False)
    # KX = linear_kernel.fit_transform(G_1)
    # KY = linear_kernel.fit_transform(G_2)
    # KXY = linear_kernel.fit_transform(G_1, G_2)

    # 2. Graph kernels
    # 2.1 WLKernel graph -> directly on graph structure
    graphs = load_graphs(
        CACHE_DIR / "sample_human_proteome_alpha_fold", graph_type="knn_graph"
    )
    KX = compute_naive_kernel(graphs)

    # 2. W-L histogram computation speedup
    KX = compute_precomp_kernel(proteins)
    # print(hashes)


if __name__ == "__main__":
    main()
