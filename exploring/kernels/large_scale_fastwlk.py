#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""large_scale_fastwlk.py

Large scale fastwlk test.

"""

import os
import random
from datetime import datetime
from tkinter import N

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from fastwlk.kernel import WeisfeilerLehmanKernel
from grakel.graph_kernels import WeisfeilerLehman
from gtda import pipeline
from tqdm import tqdm

from proteinggnnmetrics.descriptors import DegreeHistogram
from proteinggnnmetrics.distance import (
    MaximumMeanDiscrepancy,
    PearsonCorrelation,
    SpearmanCorrelation,
)
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph, KNNGraph
from proteinggnnmetrics.kernels import LinearKernel
from proteinggnnmetrics.loaders import (
    list_pdb_files,
    load_descriptor,
    load_graphs,
)
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.perturbations import GaussianNoise
from proteinggnnmetrics.protein import Protein
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.functions import (
    configure,
    flatten_lists,
    networkx2grakel,
)

config = configure()

N_JOBS = int(config["COMPUTE"]["N_JOBS"])


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
    proteins = base_feature_pipeline.fit_transform(pdb_files[:100])
    graphs = load_graphs(proteins, graph_type="eps_graph")
    grakel_test(graphs)
    fastwlk_test(graphs)


if __name__ == "__main__":
    main()
