# -*- coding: utf-8 -*-

"""pdb.py

Classes and methods to extract matrices from pdb files.

"""

import os
from pathlib import Path, PosixPath
from typing import List, Tuple

import Bio
import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PPBuilder
from joblib import Parallel, delayed
from pyparsing import col
from tqdm import tqdm

from .protein import Protein
from .utils.exception import GranularityError
from .utils.functions import distribute_function, tqdm_joblib


class Coordinates:
    """Coordinates handles coordinate extraction of atoms in pdb files."""

    def __init__(
        self, granularity: str, n_jobs: int, verbose: bool = False
    ) -> None:
        """Initializes Coordinates object

        Args:
            granularity (str, optional): granularity of the desired set of
            coordinates. Can be N, CA, C, O, or all. Defaults to "CA".
            n_jobs (int, optional): [description].
        """
        self.granularity = granularity
        self.n_jobs = n_jobs
        self.verbose = verbose

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
            name=protein_name,
            path=fname,
            coordinates=coordinates,
            sequence=sequence,
        )

    def fit(self):
        """required for sklearn compatibility"""
        pass

    def transform(self):
        """required for sklearn compatibility"""
        pass

    def fit_transform(
        self, fname_list: List[PosixPath], y=None
    ) -> List[Protein]:
        """Transform a set of pdb files to get their associated coordinates.

        Args:
            fname_list (List[PosixPath]): [description]
            granularity (str, optional): [description]. Defaults to "CA".

        Returns:
            List: list of arrays containing the coordinates for each file in fname_list
        """

        proteins = distribute_function(
            self.get_atom_coordinates,
            fname_list,
            self.n_jobs,
            "Extracting coordinates from pdb files",
            show_tqdm=self.verbose,
        )

        return proteins


class RachmachandranAngles:
    def __init__(self, from_pdb: bool, n_jobs: int, verbose: bool) -> None:
        self.from_pdb = from_pdb
        self.n_jobs = n_jobs
        self.verbose = verbose

    def get_angles_from_pdb(self, protein: Protein) -> Protein:
        """Assumes only one chain"""
        parser = PDBParser()
        structure = parser.get_structure(protein.path.stem, protein.path)

        angles = dict()
        for idx_model, model in enumerate(structure):
            polypeptides = PPBuilder().build_peptides(model)
            for idx_poly, poly in enumerate(polypeptides):
                angles[f"{idx_model}_{idx_poly}"] = poly.get_phi_psi_list()

        protein.phi_psi_angles = angles
        return protein

    def get_angles_from_coordinates(self, protein: Protein) -> Protein:
        raise NotImplementedError("Not implemented yet")

    def fit(self):
        pass

    def transform(self):
        ...

    def fit_transform(self, proteins: List[Protein], y=None):
        """Gets the angles from the list of pdb files"""

        if self.from_pdb:
            proteins = distribute_function(
                self.get_angles_from_pdb,
                proteins,
                self.n_jobs,
                "Extracting Rachmachandran angles from pdb files",
                show_tqdm=self.verbose,
            )
        else:
            proteins = distribute_function(
                self.get_angles_from_coordinates,
                proteins,
                self.n_jobs,
                "Extracting Rachmachandran angles from coordinates",
                show_tqdm=self.verbose,
            )

        return proteins
