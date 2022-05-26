#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""pipeline_example.py

Here we want to showcase the use of sklearn.pipeline to pipe operations in order to get an MMD value.

"""

import pickle
import random

import hydra
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here

from proteinmetrics.descriptors import DegreeHistogram, TopologicalDescriptor
from proteinmetrics.distance import MaximumMeanDiscrepancy
from proteinmetrics.graphs import ContactMap, KNNGraph
from proteinmetrics.kernels import (
    PersistenceFisherKernel,
    WeisfeilerLehmanGrakel,
)
from proteinmetrics.loaders import list_pdb_files, load_descriptor, load_graphs
from proteinmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates


@hydra.main(config_path=str(here()) + "/conf", config_name="conf")
def main(cfg: DictConfig):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)

    if cfg.debug.reduce_data:
        pdb_files = random.sample(pdb_files, 100)

    half = int(len(pdb_files) / 2)

    feature_pipeline = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=cfg.compute.n_jobs),
        ),
        (
            "contact map",
            ContactMap(metric="euclidean", n_jobs=cfg.compute.n_jobs),
        ),
        ("knn_graph", KNNGraph(n_neighbors=4, n_jobs=cfg.compute.n_jobs)),
        (
            "degree histogram",
            DegreeHistogram("knn_graph", n_bins=30, n_jobs=cfg.compute.n_jobs),
        ),
        (
            "tda",
            TopologicalDescriptor(
                "diagram",
                epsilon=0.01,
                n_bins=100,
                order=2,
                n_jobs=cfg.compute.n_jobs,
                landscape_layers=1,
            ),
        ),
    ]

    feature_pipeline = pipeline.Pipeline(feature_pipeline, verbose=100)

    protein_dist_1 = feature_pipeline.fit_transform(pdb_files[:2])
    protein_dist_2 = feature_pipeline.fit_transform(pdb_files[3:5])

    # Save the feature pipeline
    with open(CACHE_DIR / "protein_dist_1.pkl", "wb") as f:
        pickle.dump(protein_dist_1, f)

    with open(CACHE_DIR / "protein_dist_2.pkl", "wb") as f:
        pickle.dump(protein_dist_2, f)

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
        kernel=PersistenceFisherKernel(n_jobs=cfg.compute.n_jobs),
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
            n_jobs=cfg.compute.n_jobs,
            precomputed=False,
            n_iter=3,
            node_label="residue",
            normalize=False,
            biased=True,
        ),  # type: ignore
    ).compute(graph_dist_1, graph_dist_2)

    mmd = MaximumMeanDiscrepancy(
        biased=False,
        squared=True,
        kernel=WeisfeilerLehmanGrakel(
            n_jobs=cfg.compute.n_jobs, n_iter=3, node_label="residue",
        ),
    ).compute(graph_dist_1, graph_dist_2)
    # print(f"MMD computed from k-nn graphs is {mmd}")
    # print(f"MMD computed from WL kernel on k-nn graphs is {mmd}")


if __name__ == "__main__":
    main()
