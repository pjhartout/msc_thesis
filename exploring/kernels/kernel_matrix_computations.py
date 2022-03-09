#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""kernel_matrix_computations.py

This is to test kernel matrix computations

"""

import os
from pathlib import Path
from typing import List

import numpy as np
from grakel.kernels.vertex_histogram import VertexHistogram
from matplotlib.pyplot import show

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


@timeit
def compute_hashes_then_kernel_vec(proteins):
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
    wl_kernel = WeisfeilerLehmanKernel(
        n_iter=10,
        base_graph_kernel=VertexHistogram,
        normalize=True,
        n_jobs=int(config["COMPUTE"]["N_JOBS"]),
        pre_computed_hash=True,
        vectorized=True,
    )
    KXY = wl_kernel.fit_transform(hashes[:50], hashes[50:])
    print(np.sum(KXY))
    return proteins


@timeit
def compute_hashes_then_kernel_unordered(proteins):
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
    wl_kernel = WeisfeilerLehmanKernel(
        n_iter=10,
        base_graph_kernel=VertexHistogram,
        normalize=True,
        n_jobs=int(config["COMPUTE"]["N_JOBS"]),
        pre_computed_hash=True,
        vectorized=False,
    )
    KXY = wl_kernel.fit_transform(hashes[:50], hashes[50:])
    print(np.sum(KXY))
    return proteins


@timeit
def compute_kernel_using_precomputed_hashes_vec(proteins):
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
        vectorized=True,
    )
    KX = wl_kernel.fit_transform(hashes[:50], hashes[50:])
    print(np.sum(KX))


@timeit
def compute_kernel_using_precomputed_hashes_unordered(proteins):
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
        vectorized=False,
    )
    KX = wl_kernel.fit_transform(hashes[:50], hashes[50:])
    print(np.sum(KX))


def main():
    proteins = load_proteins(CACHE_DIR / "sample_human_proteome_alpha_fold")

    # 2. W-L histogram computation speedup
    print(
        "### Custom Implementation *without* precomputed W-L hashes vectorized ###"
    )
    proteins = compute_hashes_then_kernel_vec(proteins)

    print(
        "### Custom Implementation *without* precomputed W-L hashes unordered ###"
    )
    proteins = compute_hashes_then_kernel_unordered(proteins)

    print(
        "### Custom Implementation *with* precomputed W-L hashes vectorized ###"
    )
    compute_kernel_using_precomputed_hashes_vec(proteins)

    print(
        "### Custom Implementation *with* precomputed W-L hashes unordered ###"
    )
    compute_kernel_using_precomputed_hashes_unordered(proteins)


if __name__ == "__main__":
    main()
