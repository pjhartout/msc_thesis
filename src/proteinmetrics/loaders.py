# -*- coding: utf-8 -*-

"""loaders.py

Convenience functions to load files into memory

"""

import os
from pathlib import Path, PosixPath
from typing import List, Union

import networkx as nx
import numpy as np

from .protein import Protein
from .utils.colors import bcolors
from .utils.exception import ProteinLoadingError
from .utils.functions import filter_pdb_files


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


def load_descriptor(
    path_or_protein: Union[PosixPath, List[Protein]],
    descriptor: str,
    graph_type: str,
) -> np.ndarray:
    if type(path_or_protein) == PosixPath:
        proteins = load_proteins(path_or_protein)
    else:
        proteins = path_or_protein

    descriptor_list = list()
    for protein in proteins:
        descriptor_list.append(protein.descriptors[graph_type][descriptor])

    return np.array(descriptor_list)


def load_graphs(
    path_or_protein: Union[PosixPath, List[Protein]], graph_type: str
) -> List[nx.Graph]:
    if type(path_or_protein) == PosixPath:
        proteins = load_proteins(path_or_protein)
    else:
        proteins = path_or_protein

    graphs = list()
    for protein in proteins:
        graphs.append(protein.graphs[graph_type])
    return graphs
