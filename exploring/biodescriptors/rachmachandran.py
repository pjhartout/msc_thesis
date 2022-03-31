#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""rachmachandran.py

Test out Rachmachandran angles extraction.

"""

from proteinggnnmetrics.descriptors import RachmachandranAngles
from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.utils.functions import configure

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
    proteins = RachmachandranAngles(
        from_pdb=True, n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(proteins)
    print(proteins[0].name)
    res_1 = proteins[0].phi_psi_angles

    # Test out angle extraction from other granularities.
    proteins = Coordinates(
        granularity="backbone", n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(pdb_files)
    proteins = RachmachandranAngles(
        from_pdb=False, n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(proteins)
    print(proteins[0].name)
    res_2 = proteins[0].phi_psi_angles
    print("Check if res_1 and res_2 are the same:", res_1 == res_2)


if __name__ == "__main__":
    main()
