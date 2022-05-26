#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""rachmachandran.py

Test out Rachmachandran angles extraction.

"""

from proteinmetrics.descriptors import RamachandranAngles
from proteinmetrics.loaders import list_pdb_files
from proteinmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates
from proteinmetrics.utils.functions import configure

config = configure()

N_JOBS = 10
REDUCE_DATA = False
VERBOSE = False


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    if REDUCE_DATA:
        pdb_files = pdb_files[:10]

    proteins = Coordinates(
        granularity="CA", n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(pdb_files)
    proteins = RamachandranAngles(
        from_pdb=True, n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(proteins)
    print(proteins[0].name)
    res_1 = proteins[0].phi_psi_angles

    # Test out angle extraction from other granularities.
    proteins = Coordinates(
        granularity="backbone", n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(pdb_files)
    proteins = RamachandranAngles(
        from_pdb=False, n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(proteins)
    print(proteins[0].name)
    res_2 = proteins[0].phi_psi_angles
    print("Check if res_1 and res_2 are the same:", res_1 == res_2)


if __name__ == "__main__":
    main()
