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
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph, KNNGraph
from proteinggnnmetrics.paths import HUMAN_PROTEOME, HUMAN_PROTEOME_CA_GRAPHS
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.utils import filter_pdb_files, tqdm_joblib


@measure_memory
@timeit
def get_coords(pdb_files):
    coord = Coordinates(granularity="CA", n_jobs=N_JOBS)
    coordinates = coord.transform(pdb_files, granularity="CA")
    return coordinates


@measure_memory
@timeit
def get_contactmap(coordinates):
    contactmap = ContactMap(metric="euclidean")
    contact_maps = contactmap.transform(coordinates)
    return contact_maps


@measure_memory
@timeit
def get_knngraphs(contact_maps):
    knngraph = KNNGraph(n_neighbors=4)
    knn_graphs = knngraph.transform(contact_maps)
    return knn_graphs


@measure_memory
@timeit
def get_epsilongraph(contact_maps):
    epsilongraph = EpsilonGraph(epsilon=2)
    epsilon_graphs = epsilongraph.transform(contact_maps)
    return epsilon_graphs


@measure_memory
@timeit
def main():
    pdb_files = filter_pdb_files(os.listdir(HUMAN_PROTEOME))
    pdb_files = [HUMAN_PROTEOME / file for file in pdb_files]
    if REDUCE_DATA:
        pdb_files = random.sample(pdb_files, 10000)

    coordinates = get_coords(pdb_files)
    print(f"Coordinates: {len(coordinates)}")
    contact_maps = get_contactmap(coordinates)
    coordinates = 0
    print(f"Contact maps: {len(contact_maps)}")
    knn_graphs = get_knngraphs(contact_maps)
    print(f"KNN graphs: {len(knn_graphs)}")
    knn_graphs = 0
    epsilon_graphs = get_epsilongraph(contact_maps)
    print(f"Epsilon graphs: {len(epsilon_graphs)}")


if __name__ == "__main__":
    main()
