# -*- coding: utf-8 -*-

"""pdb.py

Classes and methods to extract matrices from pdb files.

"""

import os
from pathlib import Path, PosixPath
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from joblib import Parallel, delayed
from pyparsing import col
from tqdm import tqdm

from .constants import N_JOBS
from .errors import GranularityError
from .protein import Protein
from .utils.utils import distribute_function, tqdm_joblib
from .utils.validation import check_fnames


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

    def get_atom_coordinates(self, fname: PosixPath) -> Protein:
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
        col_names = ["x_0", "x_1", "x_2"]

        parser = PDBParser()
        structure = parser.get_structure(fname.stem, fname)
        residues = [
            r for r in structure.get_residues() if r.get_id()[0] == " "
        ]
        coordinates = list()
        sequence = list()
        if self.granularity in ["N", "CA", "C", "O"]:
            for residue in residues:
                coordinate = residue[self.granularity].get_coord()
                name = residue.get_resname()
                coordinates.append(coordinate)
                sequence.append(name)

        elif self.granularity == "all":
            # TODO: test
            for residue in residues:
                name = residue.get_resname()
                atom_coords = list()
                for atom in residue:
                    atom_coords.append(atom.get_coord())
                    coordinates.append(atom_coords)
                    sequence.append(name)

        else:
            raise GranularityError("Specify correct granularity")

        coordinates = np.vstack(coordinates)
        protein_name = Path(fname).name.split(".")[0]
        return Protein(
            name=protein_name, coordinates=coordinates, sequence=sequence
        )

    def extract(
        self, fname_list: List[PosixPath], granularity: str = "CA"
    ) -> List[np.ndarray]:
        """Transform a set of pdb files to get their associated coordinates.

        Args:
            fname_list (List[PosixPath]): [description]
            granularity (str, optional): [description]. Defaults to "CA".

        Returns:
            List: list of arrays containing the coordinates for each file in fname_list
        """
        fname_list = check_fnames(fname_list)

        proteins = distribute_function(
            self.get_atom_coordinates,
            fname_list,
            "Extracting coordinates from pdb files",
            self.n_jobs,
        )

        return proteins
