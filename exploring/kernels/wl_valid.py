#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""filename.py

***file description***

"""

import numpy as np
from grakel import VertexHistogram
from pytest import mark

from proteinggnnmetrics.kernels import WeisfeilerLehmanKernel
from proteinggnnmetrics.loaders import load_graphs, load_proteins
from proteinggnnmetrics.paths import CACHE_DIR
from proteinggnnmetrics.utils.functions import configure, distribute_function

config = configure()


def compute_wl_hashes(protein):
    protein.set_weisfeiler_lehman_hashes(graph_type="knn_graph", n_iter=10)
    return protein


def compute_standard_kernel(proteins):
    half = int(len(proteins) / 2)

    graphs = load_graphs(proteins, graph_type="knn_graph")

    in1 = graphs[:half]
    in2 = graphs[half:]

    wl_kernel = WeisfeilerLehmanKernel(
        n_iter=5,
        normalize=True,
        n_jobs=-1,
        pre_computed_hash=False,
        base_graph_kernel=VertexHistogram,
    )
    return np.sum(wl_kernel.fit_transform(in1, in2))


def compute_fast_wl_kernel(proteins):
    half = int(len(proteins) / 2)

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
        n_iter=5,
        normalize=True,
        n_jobs=int(config["COMPUTE"]["N_JOBS"]),
        pre_computed_hash=True,
    )

    in1 = hashes[:half]
    in2 = hashes[half:]

    return np.sum(np.array(wl_kernel.fit_transform(in1, in2)))


def main():
    """we want to test this function by comparing the result of two
    implementations
    """
    proteins = load_proteins(CACHE_DIR / "sample_human_proteome_alpha_fold")
    result_1 = compute_standard_kernel(proteins)
    print(result_1)

    result_2 = compute_fast_wl_kernel(proteins)
    print(result_2)


if __name__ == "__main__":
    main()
