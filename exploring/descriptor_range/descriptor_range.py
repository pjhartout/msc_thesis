#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""descriptor_range.py

What is:
    - The range of the laplacian
    - The range of the degree?
    
max degree 719
laplacian  1.2586714829332633
distance   462.5783386230469
"""
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from gtda import pipeline

from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.loaders import list_pdb_files
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates

N_JOBS = 100

log = logging.getLogger(__name__)


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=N_JOBS, verbose=True),
        ),
        (
            "contact_map",
            ContactMap(n_jobs=N_JOBS, verbose=True),
        ),
        (
            "eps_graph",
            EpsilonGraph(epsilon=32, n_jobs=N_JOBS, verbose=True),
        ),
    ]

    proteins = pipeline.Pipeline(base_feature_steps).fit_transform(pdb_files)
    max_degree = max(
        [
            len(nx.degree_histogram(protein.graphs["eps_graph"]))
            for protein in proteins
        ]
    )
    max_laplacian = max(
        [
            max(nx.normalized_laplacian_spectrum(protein.graphs["eps_graph"]))
            for protein in proteins
        ]
    )
    max_distance = max(
        [np.max(protein.contact_map.flatten()) for protein in proteins]
    )

    print(f"max degree  {max_degree}")
    log.info(f"max degree  {max_degree}")

    print(f"laplacian  {max_laplacian}")
    log.info(f"laplacian  {max_laplacian}")

    print(f"max distance {max_distance}")
    log.info(f"max distance {max_distance}")


if __name__ == "__main__":
    main()
