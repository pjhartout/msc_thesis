#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""filename.py

***file description***

"""

from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates, RachmachandranAngles
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
    proteins = RachmachandranAngles(
        n_jobs=N_JOBS, verbose=VERBOSE
    ).fit_transform(proteins)
    print(proteins[0].phi_psi_angles)


if __name__ == "__main__":
    main()
