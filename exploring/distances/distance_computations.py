#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""distance_computations.py

Computes MMD, other distances from features extracted from protein

"""

import numpy as np

from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.kernels import LinearKernel, WeisfeilerLehmanKernel
from proteinggnnmetrics.loaders import (
    load_descriptor,
    load_graphs,
    load_proteins,
)
from proteinggnnmetrics.paths import CACHE_DIR
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.functions import distribute_function


def compute_wl_hashes(protein):
    protein.set_weisfeiler_lehman_hashes(graph_type="knn_graph", n_iter=10)
    return protein


def main():
    proteins = load_proteins(CACHE_DIR / "sample_human_proteome_alpha_fold")
    half = int(len(proteins) / 2)

    # Linear kernel, degree dist
    degree_histograms = load_descriptor(
        proteins, descriptor="degree_histogram", graph_type="knn_graph",
    )
    linear_kernel = LinearKernel(dense_output=False)
    mmd = MaximumMeanDiscrepancy(kernel=linear_kernel)
    result = mmd.fit_transform(
        np.array(degree_histograms), np.array(degree_histograms)
    )
    print(result)
    # # Graph kernel, W-L
    # proteins = distribute_function(
    #     compute_wl_hashes,
    #     proteins,
    #     int(config["COMPUTE"]["N_JOBS"]),
    #     "Computing Weisfeiler-Lehman Hashes",
    # )
    # hashes = [
    #     protein.descriptors["knn_graph"]["weisfeiler-lehman-hist"]
    #     for protein in proteins
    # ]
    # wl_kernel = WeisfeilerLehmanKernel(
    #     n_iter=5,
    #     normalize=True,
    #     n_jobs=int(config["COMPUTE"]["N_JOBS"]),
    #     pre_computed_hash=True,
    # )
    # mmd = MaximumMeanDiscrepancy(kernel=wl_kernel)
    # result = mmd.fit_transform(hashes[:half], hashes[half:])
    # print(result)


if __name__ == "__main__":
    main()
