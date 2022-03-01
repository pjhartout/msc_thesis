#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""kernel_matrix_computations.py

This is to test kernel matrix computations

"""

import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from grakel.kernels import VertexHistogram, WeisfeilerLehman

from proteinggnnmetrics.kernels import LinearKernel, WLKernel
from proteinggnnmetrics.loaders import (
    load_descriptor,
    load_graphs,
    load_proteins,
)
from proteinggnnmetrics.paths import CACHE_DIR
from proteinggnnmetrics.protein import Protein


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
    wl_kernel = WLKernel()

    G_dist_1 = graphs[half_point:]
    G_dist_2 = graphs[:half_point]
    KX = wl_kernel.transform(G_dist_1)
    KY = wl_kernel.transform(G_dist_2)
    KXY = wl_kernel.transform(G_dist_1, G_dist_2)


if __name__ == "__main__":
    main()
