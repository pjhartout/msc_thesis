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
from joblib import Parallel, delayed

from proteinggnnmetrics.constants import N_JOBS, REDUCE_DATA
from proteinggnnmetrics.graphs.extraction import ContactMap
from proteinggnnmetrics.paths import HUMAN_PROTEOME, HUMAN_PROTEOME_CA_GRAPHS
from proteinggnnmetrics.utils import filter_pdb_files, timeit


@timeit
def compute_contact_map_dist_task(contactmap, files):
    res = list()
    for file in files:
        res.append(cm_transform(contactmap, HUMAN_PROTEOME / file, N_JOBS))
    return res


@timeit  # Faster
def compute_contact_map_dist_files(contactmap, files):
    res = Parallel(n_jobs=N_JOBS)(
        delayed(cm_transform)(contactmap, HUMAN_PROTEOME / file, 1)
        for file in files
    )
    return res


def cm_transform(contactmap, file, n_jobs):
    contact_map = contactmap.transform(file, n_jobs)
    return contact_map


def main():
    random_state = np.random.RandomState(42)

    pdb_files = filter_pdb_files(os.listdir(HUMAN_PROTEOME))

    if REDUCE_DATA:
        pdb_files = random.sample(pdb_files, 1000)

    contactmap = ContactMap(granularity="CA", random_state=random_state)
    res_1 = compute_contact_map_dist_files(contactmap, pdb_files)
    res_2 = compute_contact_map_dist_task(contactmap, pdb_files)


if __name__ == "__main__":
    main()
