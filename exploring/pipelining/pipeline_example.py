#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""pipeline_example.py

Here we want to showcase the use of sklearn.pipeline to pipe operations in order to get an MMD value.

"""

import pickle
import random

import numpy as np
import pandas as pd
from fastwlk.kernel import WeisfeilerLehmanKernel
from grakel import WeisfeilerLehman
from gtda import pipeline
from numpy import square

from proteinggnnmetrics.descriptors import (
    DegreeHistogram,
    TopologicalDescriptor,
)
from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.graphs import ContactMap, KNNGraph
from proteinggnnmetrics.kernels import (
    LinearKernel,
    PersistenceFisherKernel,
    WeisfeilerLehmanGrakel,
)
from proteinggnnmetrics.loaders import (
    list_pdb_files,
    load_descriptor,
    load_graphs,
)
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.utils.functions import configure

config = configure()

N_JOBS = int(config["COMPUTE"]["N_JOBS"])
REDUCE_DATA = config["DEBUG"]["REDUCE_DATA"]


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)

    if REDUCE_DATA:
        pdb_files = random.sample(pdb_files, 100)

    half = int(len(pdb_files) / 2)

    feature_pipeline = [
        ("coordinates", Coordinates(granularity="CA", n_jobs=N_JOBS),),
        ("contact map", ContactMap(metric="euclidean", n_jobs=N_JOBS)),
        ("knn_graph", KNNGraph(n_neighbors=4, n_jobs=N_JOBS)),
        (
            "degree histogram",
            DegreeHistogram("knn_graph", n_bins=30, n_jobs=N_JOBS),
        ),
        (
            "tda",
            TopologicalDescriptor(
                "diagram",
                epsilon=0.01,
                n_bins=100,
                order=2,
                n_jobs=N_JOBS,
                landscape_layers=1,
            ),
        ),
    ]

    feature_pipeline = pipeline.Pipeline(feature_pipeline, verbose=100)

    # protein_dist_1 = feature_pipeline.fit_transform(pdb_files[:10])
    # protein_dist_2 = feature_pipeline.fit_transform(pdb_files[10:20])

    # # Save the feature pipeline
    # with open(CACHE_DIR / "protein_dist_1.pkl", "wb") as f:
    #     pickle.dump(protein_dist_1, f)

    # with open(CACHE_DIR / "protein_dist_2.pkl", "wb") as f:
    #     pickle.dump(protein_dist_2, f)

    # Caching to accelerate debugging
    # Save protein_dist_1 and protein_dist_2
    with open(CACHE_DIR / "protein_dist_1.pkl", "rb") as f:
        protein_dist_1 = pickle.load(f)
    # take 10 samples
    protein_dist_1 = protein_dist_1[:10]

    with open(CACHE_DIR / "protein_dist_2.pkl", "rb") as f:
        protein_dist_2 = pickle.load(f)
    # take 11 samples
    protein_dist_2 = protein_dist_2[:11]

    dist_1 = load_descriptor(protein_dist_1, "diagram", "contact_graph")
    dist_2 = load_descriptor(protein_dist_2, "diagram", "contact_graph")

    mmd = MaximumMeanDiscrepancy(
        biased=False,
        squared=True,
        kernel=PersistenceFisherKernel(n_jobs=N_JOBS),
    ).compute(dist_1, dist_2)

    # dist_1 = load_descriptor(protein_dist_1, "degree_histogram", "knn_graph")
    # dist_2 = load_descriptor(protein_dist_2, "degree_histogram", "knn_graph")

    # mmd = MaximumMeanDiscrepancy(
    #     kernel=LinearKernel(dense_output=False)
    # ).compute(dist_1, dist_2)

    # print(f"MMD computed from degree histograms on k-nn graphs is {mmd}")

    graph_dist_1 = load_graphs(protein_dist_1, "knn_graph")
    graph_dist_2 = load_graphs(protein_dist_2, "knn_graph")

    mmd = MaximumMeanDiscrepancy(
        biased=False,
        squared=True,
        kernel=WeisfeilerLehmanKernel(
            n_jobs=N_JOBS,
            precomputed=False,
            n_iter=3,
            node_label="residue",
            normalize=False,
            biased=True,
        ),
    ).compute(graph_dist_1, graph_dist_2)

    mmd = MaximumMeanDiscrepancy(
        biased=False,
        squared=True,
        kernel=WeisfeilerLehmanGrakel(
            n_jobs=N_JOBS, n_iter=3, node_label="residue",
        ),
    ).compute(graph_dist_1, graph_dist_2)
    # print(f"MMD computed from k-nn graphs is {mmd}")
    # print(f"MMD computed from WL kernel on k-nn graphs is {mmd}")


if __name__ == "__main__":
    main()
