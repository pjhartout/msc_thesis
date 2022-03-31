#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""distance_distribution.py

Tests out clashings and distance distribution.

"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from proteinggnnmetrics.graphs import ContactMap
from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates, RachmachandranAngles


N_JOBS = 10
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

    distances = [protein.contact_map.flatten() for protein in proteins]
    # Plot histogram of distances.
    sns.set()
    plt.xlabel("Distance (Angstrom)")
    plt.ylabel("Count")
    plt.title("Distance Distributions of the Alpha Carbon Atoms")
    plt.hist(
        np.concatenate(distances),
        bins=500,
        density=True,
        facecolor="g",
        histtype="stepfilled",
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
