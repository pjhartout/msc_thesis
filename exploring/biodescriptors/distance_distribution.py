#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""distance_distribution.py

Tests out clashings and distance distribution.

"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from proteinggnnmetrics.descriptors import DistanceHistogram
from proteinggnnmetrics.graphs import ContactMap
from proteinggnnmetrics.kernels import LinearKernel
from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates

N_JOBS = 4
REDUCE_DATA = True
VERBOSE = False


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    if REDUCE_DATA:
        pdb_files = pdb_files[:10]

    proteins = Coordinates(
        granularity="CA", n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(pdb_files)
    proteins = ContactMap(metric="euclidean", n_jobs=N_JOBS).fit_transform(
        proteins
    )

    proteins = DistanceHistogram(
        n_bins=100, n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(proteins)

    hists = np.array([protein.distance_hist for protein in proteins])
    lk = LinearKernel(n_jobs=N_JOBS, verbose=VERBOSE)
    res = lk.compute_matrix(hists)
    print(res.shape)


if __name__ == "__main__":
    main()
