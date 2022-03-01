#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""loaders.py

Convenience functions to load files into memory

"""

import os
from pathlib import Path, PosixPath
from typing import List

from .errors import ProteinLoadingError
from .protein import Protein
from .utils.colors import bcolors
from .utils.utils import filter_pdb_files


def list_pdb_files(path: PosixPath) -> List[PosixPath]:
    """Lists pdb files in given path"""
    pdb_files = filter_pdb_files(os.listdir(path))
    return [Path(path) / pdb_file for pdb_file in pdb_files]


def load_proteins(path: PosixPath) -> List[Protein]:
    print(bcolors.OKBLUE + f"Loading proteins from {path}" + bcolors.ENDC)
    files = os.listdir(path)
    prot = Protein()
    proteins = list()
    for pkl_file in files:
        if ".pkl" in pkl_file:
            try:
                proteins.append(prot.load(path / pkl_file))
            except:
                raise ProteinLoadingError(f"Failed to load {path}/{pkl_file}")
        else:
            print(
                bcolors.WARNING + f"Skipped {path}/{pkl_file}" + bcolors.ENDC
            )

    return proteins


def load_descriptor(path: PosixPath, descriptor: str, graph_type: str):
    proteins = load_proteins(path)
    degree_histograms = list()
    for protein in proteins:
        degree_histograms.append(protein.descriptors[graph_type][descriptor])
    return degree_histograms


def load_graphs(path: PosixPath, graph_type: str):
    proteins = load_proteins(path)
    graphs = list()
    for protein in proteins:
        graphs.append(protein.graphs[graph_type])
    return graphs
