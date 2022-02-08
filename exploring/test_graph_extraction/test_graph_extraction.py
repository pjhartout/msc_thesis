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
from tqdm import tqdm

from proteinggnnmetrics.constants import N_JOBS, REDUCE_DATA
from proteinggnnmetrics.debug import timeit
from proteinggnnmetrics.graphs import (
    ContactMap,
    EpsilonGraph,
    KNNGraph,
    ParallelGraphExtraction,
)
from proteinggnnmetrics.paths import HUMAN_PROTEOME, HUMAN_PROTEOME_CA_GRAPHS
from proteinggnnmetrics.pdb import ParallelCoordinates
from proteinggnnmetrics.utils import filter_pdb_files, tqdm_joblib


def cm_transform(contactmap, file, n_jobs):
    contact_map = contactmap.transform(file, n_jobs)
    return contact_map


@timeit
def main():

    pdb_files = filter_pdb_files(os.listdir(HUMAN_PROTEOME))
    pdb_files = [HUMAN_PROTEOME / file for file in pdb_files]
    if REDUCE_DATA:
        pdb_files = random.sample(pdb_files, 1000)

    parallel_cooords = ParallelCoordinates(n_jobs=N_JOBS)
    coordinates = parallel_cooords.get_coordinates_from_files(
        pdb_files, granularity="CA"
    )

    contactmap = ContactMap(metric="euclidean", n_jobs=1)
    parallelextraction = ParallelGraphExtraction(n_jobs=N_JOBS)
    contact_maps = parallelextraction.transform(
        coordinates, contactmap.transform
    )

    knngraph = KNNGraph(n_neighbors=4)
    knn_graphs = parallelextraction.transform(contact_maps, knngraph.transform)

    epsilongraph = EpsilonGraph(epsilon=6)
    epsilon_graphs = parallelextraction.transform(
        contact_maps, epsilongraph.transform
    )

    plt.imshow(knn_graphs[0].toarray(), interpolation="nearest")
    plt.savefig("sample_knn.png")

    plt.imshow(epsilon_graphs[0], interpolation="nearest")
    plt.savefig("sample_epsilon.png")


if __name__ == "__main__":
    main()
