#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pdb2contact_map.py

This file converts *.pdb files to contact maps in parallel.

"""

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from joblib import Parallel, delayed
from tqdm import tqdm

from gnn_metrics.constants import N_JOBS, REDUCE_DATA
from gnn_metrics.paths import HUMAN_PROTEOME, HUMAN_PROTEOME_CA_CONTACT_MAP
from gnn_metrics.utils import filter_monomers, filter_pdb_files, make_dir

parser = PDBParser()


def pdb2contact_map(file: str, granularity: str = "CA"):
    """Transforms a pdb file to a contact map following the granularity parameter

    Args:
        granularity (str): granularity parameter used to extract the graph.
            Defaults to "CA" (alpha carbon atom).
        file (str): file name of the pdb to extract the graph from
    """

    file_path = Path(file)
    structure = parser.get_structure(
        file_path.stem, HUMAN_PROTEOME / file_path
    )

    residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
    contact_map = np.zeros((len(residues), len(residues)))
    for x in tqdm(range(len(residues))):
        for y in range(len(residues)):
            one = residues[x][granularity].get_coord()
            two = residues[y][granularity].get_coord()
            contact_map[x, y] = np.linalg.norm(one - two)
    with open(
        str(HUMAN_PROTEOME_CA_CONTACT_MAP / file_path.stem) + ".npy", "wb"
    ) as f:
        np.save(f, contact_map)


def main():
    pdb_files = filter_pdb_files(os.listdir(HUMAN_PROTEOME))
    pdb_files = filter_monomers(
        [str(HUMAN_PROTEOME / Path(file)) for file in pdb_files]
    )
    make_dir(HUMAN_PROTEOME_CA_CONTACT_MAP)

    if REDUCE_DATA:
        pdb_files = random.sample(pdb_files, 1000)

    Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(pdb2contact_map)(file) for file in pdb_files
    )


if __name__ == "__main__":
    main()
