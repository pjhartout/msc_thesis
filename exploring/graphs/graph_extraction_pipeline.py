#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""graph_extraction_pipeline.py

This is test benchmark to test out the graph extraction process on alphafold data.

"""

import os
import random

from tqdm import tqdm

from proteinggnnmetrics.descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    LaplacianSpectrum,
    TopologicalDescriptor,
)
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph, KNNGraph
from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.utils.debug import measure_memory, timeit

random.seed(42)

N_JOBS = 10
REDUCE_DATA = True


@measure_memory
@timeit
def get_coords(pdb_files):
    coord = Coordinates(granularity="CA", n_jobs=N_JOBS)
    proteins = coord.fit_transform(pdb_files)
    return proteins


@measure_memory
@timeit
def get_contactmaps(proteins):
    contactmap = ContactMap(metric="euclidean", n_jobs=N_JOBS)
    proteins = contactmap.fit_transform(proteins)
    return proteins


@measure_memory
@timeit
def get_knngraphs(proteins):
    knngraph = KNNGraph(n_neighbors=4, n_jobs=N_JOBS)
    proteins = knngraph.fit_transform(proteins)
    return proteins


@measure_memory
@timeit
def get_epsilongraphs(proteins):
    epsilongraph = EpsilonGraph(epsilon=2, n_jobs=N_JOBS)
    proteins = epsilongraph.fit_transform(proteins)
    return proteins


@measure_memory
@timeit
def get_deg_histograms(proteins):
    degree_histogram = DegreeHistogram("knn_graph", n_bins=30, n_jobs=N_JOBS)
    proteins = degree_histogram.fit_transform(proteins)
    return proteins


@measure_memory
@timeit
def get_clu_histograms(proteins):
    clustering_histogram = ClusteringHistogram("knn_graph", n_jobs=N_JOBS)
    proteins = clustering_histogram.fit_transform(proteins)
    return proteins


@measure_memory
@timeit
def get_spectrum(proteins):
    laplancian_histogram = LaplacianSpectrum(
        "knn_graph", n_bins=30, n_jobs=N_JOBS
    )
    proteins = laplancian_histogram.fit_transform(proteins)
    return proteins


@measure_memory
@timeit
def get_tda_descriptor(proteins):
    tda_descriptor = TopologicalDescriptor(
        "landscape",
        epsilon=0.01,
        n_bins=100,
        order=1,
        n_jobs=N_JOBS,
        landscape_layers=1,
    )
    proteins = tda_descriptor.fit_transform(proteins)
    return proteins


@measure_memory
@timeit
def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)

    if REDUCE_DATA:
        pdb_files = random.sample(pdb_files, 100)

    proteins = get_coords(pdb_files)
    print(f"Coordinates: {len(proteins)}")
    proteins = get_contactmaps(proteins)
    proteins = get_knngraphs(proteins)
    proteins = get_epsilongraphs(proteins)
    proteins = get_deg_histograms(proteins)
    proteins = get_clu_histograms(proteins)
    # proteins = get_spectrum(proteins)
    proteins = get_tda_descriptor(proteins)
    for protein in tqdm(proteins):
        protein.save(CACHE_DIR / "sample_human_proteome_alpha_fold")
    print("END CALLS")


if __name__ == "__main__":
    main()
