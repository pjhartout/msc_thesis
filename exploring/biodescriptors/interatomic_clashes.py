#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""interatomic_clashes.py

Test out the calculations the number of interatomic clashes for each protein.

"""

from proteinggnnmetrics.descriptors import InteratomicClash
from proteinggnnmetrics.graphs import ContactMap
from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.utils.functions import configure

config = configure()

N_JOBS = int(config["COMPUTE"]["N_JOBS"])
REDUCE_DATA = config["DEBUG"]["REDUCE_DATA"]
VERBOSE = config["RUNTIME"]["VERBOSE"]


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    if REDUCE_DATA:
        pdb_files = pdb_files[:10]

    proteins = Coordinates(
        granularity="CA", n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(pdb_files)
    proteins = ContactMap(n_jobs=N_JOBS, verbose=VERBOSE).fit_transform(
        proteins
    )
    proteins = InteratomicClash(
        threshold=1.5, n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(proteins)
    print(
        f"Ratio of interatomic clashes with low threshold{proteins[0].interatomic_clashes}"
    )

    proteins = InteratomicClash(
        threshold=4, n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(proteins)
    print(
        f"Ratio of interatomic clashes with intermediate threshold{proteins[0].interatomic_clashes}"
    )

    proteins = InteratomicClash(
        threshold=400, n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(proteins)
    print(
        f"Ratio of interatomic clashes with high threshold{proteins[0].interatomic_clashes}"
    )


if __name__ == "__main__":
    main()
