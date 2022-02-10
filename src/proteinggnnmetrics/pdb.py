# -*- coding: utf-8 -*-

"""pdb.py

Classes and methods to extract matrices from pdb files.

"""

import os
from pathlib import Path, PosixPath
from typing import List

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from joblib import Parallel, delayed
from tqdm import tqdm

from proteinggnnmetrics.errors import GranularityError
from proteinggnnmetrics.utils.utils import tqdm_joblib
from proteinggnnmetrics.utils.validation import check_fnames

from .constants import N_JOBS
from .utils.utils import tqdm_joblib


class Coordinates:
    """Coordinates handles coordinate extraction of atoms in pdb files."""

    def __init__(self, granularity: str = "CA", n_jobs: int = N_JOBS) -> None:
        """Initializes Coordinates object

        Args:
            granularity (str, optional): granularity of the desired set of
            coordinates. Can be N, CA, C, O, or all. Defaults to "CA".
            n_jobs (int, optional): [description]. Defaults to N_JOBS.
        """
        self.granularity = granularity
        self.n_jobs = n_jobs

    def get_atom_coordinates(self, fname: PosixPath) -> np.ndarray:
        """Given a file name, extracts its atom coordinates. self.granularity
        determines which atoms are considered.

        Args:
            fname (PosixPath): path to the file to extract coordinates from.

        Raises:
            GranularityError: error raised when granularity is set incorrectly.

        Returns:
            np.ndarray: array of coordinates of each of the atoms in fname
            (shape: n_atoms, 3)
        """
        parser = PDBParser()
        structure = parser.get_structure(fname.stem, fname)
        residues = [
            r for r in structure.get_residues() if r.get_id()[0] == " "
        ]
        coordinates = list()
        if self.granularity in ["N", "CA", "C", "O"]:
            for x in range(len(residues)):
                coordinates.append(residues[x][self.granularity].get_coord())
        elif self.granularity == "all":  # untested
            for residue in range(len(residues)):
                for atom in residue:
                    coordinates.append(atom.get_coord())
        else:
            raise GranularityError("Specify correct granularity")

        return np.vstack(coordinates)

    def transform(
        self, fname_list: List[str], granularity: str = "CA"
    ) -> List[np.ndarray]:
        """Transform a set of pdb files to get their associated coordinates.

        Args:
            fname_list (List[str]): [description]
            granularity (str, optional): [description]. Defaults to "CA".

        Returns:
            List: list of arrays containing the coordinates for each file in fname_list
        """
        fname_list = check_fnames(fname_list)
        with tqdm_joblib(
            tqdm(
                desc="Extracting coordinates from pdb files",
                total=len(fname_list),
            )
        ) as progressbar:
            Xt = Parallel(n_jobs=self.n_jobs)(
                delayed(self.get_atom_coordinates)(file) for file in fname_list
            )
        return Xt
