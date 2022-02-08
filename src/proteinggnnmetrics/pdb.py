# -*- coding: utf-8 -*-

"""pdb.py

PDB helper functions

TODO: docstrings, type hints.
"""

import os
from pathlib import Path, PosixPath
from typing import List

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from joblib import Parallel, delayed
from tqdm import tqdm

from proteinggnnmetrics.utils import tqdm_joblib

from .errors import FileExtentionError
from .paths import HUMAN_PROTEOME, HUMAN_PROTEOME_CA_GRAPHS
from .utils import tqdm_joblib


class Coordinates:
    def __init__(self, granularity="CA") -> None:
        self.granularity = granularity

    def get_atom_coordinates(self, fname: PosixPath) -> np.ndarray:
        parser = PDBParser()
        structure = parser.get_structure(fname.stem, fname)
        residues = [
            r for r in structure.get_residues() if r.get_id()[0] == " "
        ]
        coordinates = list()
        for x in range(len(residues)):
            coordinates.append(residues[x][self.granularity].get_coord())

        return np.vstack(coordinates)


class ParallelCoordinates:
    """
    Wrapper of Coordinates to extract coordinates in separate threads to
    process multiple files simultaneously. Useful when dealing with
    proteome-scale datasets.
    """

    def __init__(self, n_jobs) -> None:
        self.n_jobs = n_jobs

    def get_coordinates_from_files(
        self, fname_list: List[str], granularity="CA"
    ) -> np.ndarray:

        pathname_list = [Path(file) for file in fname_list]
        for file in pathname_list:
            if file.suffix != ".pdb":
                raise FileExtentionError(
                    "Make sure you're only processing PDB files!"
                )
        coordinates = Coordinates(granularity=granularity)
        with tqdm_joblib(
            tqdm(
                desc="Extracting coordinates from pdb files",
                total=len(pathname_list),
            )
        ) as progressbar:
            fname_list_coordinates = Parallel(n_jobs=self.n_jobs)(
                delayed(coordinates.get_atom_coordinates)(file)
                for file in pathname_list
            )
        return fname_list_coordinates
