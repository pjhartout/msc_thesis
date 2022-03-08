#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""kernel_matrix_computations.py

This is to test kernel matrix computations

"""

import os
from pathlib import Path
from typing import List

from grakel.kernels.vertex_histogram import VertexHistogram

from proteinggnnmetrics.kernels import LinearKernel, WeisfeilerLehmanKernel
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
    protein.set_weisfeiler_lehman_hashes(graph_type="knn_graph", n_iter=10)
    return protein


@measure_memory
@timeit
def compute_naive_kernel(graphs):
    wl_kernel = WeisfeilerLehmanKernel(
        n_iter=10,
        base_graph_kernel=VertexHistogram,
        normalize=True,
        n_jobs=int(config["COMPUTE"]["N_JOBS"]),
        pre_computed_hash=False,
    )
    KX = wl_kernel.fit_transform(graphs)
    return KX


@measure_memory
@timeit
def compute_hashes_then_kernel(proteins):
    proteins = distribute_function(
        compute_wl_hashes,
        proteins,
        int(config["COMPUTE"]["N_JOBS"]),
        "Computing Weisfeiler-Lehman Hashes",
    )
    hashes = [
        protein.descriptors["knn_graph"]["weisfeiler-lehman-hist"]
        for protein in proteins
    ]
    wl_kernel = WeisfeilerLehmanKernel(
        n_iter=10,
        base_graph_kernel=VertexHistogram,
        normalize=True,
        n_jobs=int(config["COMPUTE"]["N_JOBS"]),
        pre_computed_hash=False,
    )
    KX = wl_kernel.fit_transform(hashes)
    print("Done")
    return KX, proteins


@measure_memory
@timeit
def compute_kernel_using_precomputed_hashes(proteins):
    hashes = [
        protein.descriptors["knn_graph"]["weisfeiler-lehman-hist"]
        for protein in proteins
    ]
    wl_kernel = WeisfeilerLehmanKernel(
        n_iter=10,
        base_graph_kernel=VertexHistogram,
        normalize=True,
        n_jobs=int(config["COMPUTE"]["N_JOBS"]),
        pre_computed_hash=True,
    )
    KX = wl_kernel.fit_transform(hashes)
    print("Done")
    return KX


def main():

    proteins = load_proteins(CACHE_DIR / "sample_human_proteome_alpha_fold")

    # 1. Generic kernel
    # 1.1 Linear kernel
    # degree_histograms = load_descriptor(
    #     CACHE_DIR / "sample_human_proteome_alpha_fold",
    #     descriptor="degree_histogram",
    #     graph_type="knn_graph",
    # )
    # linear_kernel = LinearKernel(dense_output=False,)
    # KX = linear_kernel.transform(degree_histograms)

    # 2. Graph kernels
    # 2.1 WLKernel graph -> directly on graph structure
    # print("### Grakel Implementation ###")
    # graphs = load_graphs(proteins, graph_type="knn_graph")
    # KX = compute_naive_kernel(graphs)

    # 2. W-L histogram computation speedup
    print("### Custom Implementation *without* precomputed W-L hashes ###")
    KX, proteins = compute_hashes_then_kernel(proteins)

    print("### Custom Implementation *with* precomputed W-L hashes ###")
    KX = compute_kernel_using_precomputed_hashes(proteins)
    # print(hashes)


if __name__ == "__main__":
    main()
