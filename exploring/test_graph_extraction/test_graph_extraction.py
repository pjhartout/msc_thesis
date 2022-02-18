#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_graph_extraction.py

This is test benchmark to test out the graph extraction process on alphafold data.

"""

import os
import random
from pathlib import Path
from pickle import dump

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from joblib import Parallel, delayed
from scipy import sparse
from tqdm import tqdm

from proteinggnnmetrics.constants import N_JOBS, REDUCE_DATA
from proteinggnnmetrics.descriptors import DegreeHistogram
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph, KNNGraph
from proteinggnnmetrics.paths import HUMAN_PROTEOME, HUMAN_PROTEOME_CA_GRAPHS
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.utils import filter_pdb_files


@measure_memory
@timeit
def get_coords(pdb_files):
    coord = Coordinates(granularity="CA", n_jobs=N_JOBS)
    proteins = coord.extract(pdb_files, granularity="CA")
    return proteins


@measure_memory
@timeit
def get_contactmaps(proteins):
    contactmap = ContactMap(metric="euclidean")
    proteins = contactmap.construct(proteins)
    return proteins


@measure_memory
@timeit
def get_knngraphs(proteins):
    knngraph = KNNGraph(n_neighbors=4)
    proteins = knngraph.construct(proteins)
    return proteins


@measure_memory
@timeit
def get_epsilongraphs(proteins):
    epsilongraph = EpsilonGraph(epsilon=2)
    proteins = epsilongraph.construct(proteins)
    return proteins


@measure_memory
@timeit
def get_histograms(proteins):
    degree_histogram = DegreeHistogram("knn_graph", hist_len=30, n_jobs=N_JOBS)
    histograms = degree_histogram.describe(proteins)
    return histograms


@measure_memory
@timeit
def main():
    pdb_files = filter_pdb_files(os.listdir(HUMAN_PROTEOME))
    pdb_files = [HUMAN_PROTEOME / file for file in pdb_files]
    if REDUCE_DATA:
        pdb_files = random.sample(pdb_files, 10)

    proteins = get_coords(pdb_files)
    print(f"Coordinates: {len(proteins)}")
    proteins = get_contactmaps(proteins)
    proteins = get_knngraphs(proteins)
    proteins = get_epsilongraphs(proteins)

    histograms = get_histograms(proteins)
    print("END DEBUG")


if __name__ == "__main__":
    main()
