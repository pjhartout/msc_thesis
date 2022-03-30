#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""mmd_testing.py

Computes MMD test between two distributions

"""

import pickle
from collections import Counter
from pathlib import Path, PosixPath
from typing import Callable, Dict, List

import fastwlk
import numpy as np
import scipy
from grakel import WeisfeilerLehman
from gtda import pipeline
from fastwlk.kernel import WeisfeilerLehmanKernel
from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.kernels import Kernel
from proteinggnnmetrics.loaders import list_pdb_files, load_graphs
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.statistics import MMDTest

N_JOBS = 10


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    base_feature_steps = [
        ("coordinates", Coordinates(granularity="CA", n_jobs=N_JOBS)),
        ("contact map", ContactMap(metric="euclidean", n_jobs=N_JOBS)),
        ("epsilon graph", EpsilonGraph(epsilon=4, n_jobs=N_JOBS)),
    ]
    base_feature_pipeline = pipeline.Pipeline(
        base_feature_steps, verbose=False
    )
    proteins = base_feature_pipeline.fit_transform(pdb_files[:100])
    dist_1 = load_graphs(proteins[:60], "eps_graph")
    dist_2 = load_graphs(proteins[60:], "eps_graph")

    p_value = MMDTest(
        alpha=0.05, m=20, t=1000, kernel=WeisfeilerLehmanKernel(n_jobs=N_JOBS)
    ).compute_p_value(dist_1, dist_2)


if __name__ == "__main__":
    main()
