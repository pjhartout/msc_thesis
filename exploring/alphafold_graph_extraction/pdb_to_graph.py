#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pdb_to_graph.py

This file converts *.pdb files to pickled networkx files in parallel.

"""
import os
from pathlib import Path
from pickle import dump

import matplotlib.pyplot as plt
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from joblib import Parallel, delayed

from gnn_metrics.constants import N_JOBS
from gnn_metrics.paths import HUMAN_PROTEOME, HUMAN_PROTEOME_CA_GRAPHS
from gnn_metrics.utils import filter_monomers, filter_pdb_files, make_dir


def pdb_to_graph(file: str, granularity: str = "CA"):
    """Transforms a pdb file to a graph following the granularity parameter

    Args:
        granularity (str): granularity parameter used to extract the graph.
            Defaults to "CA" (alpha carbon atom).
        file (str): file name of the pdb to extract the graph from
    """
    c = ProteinGraphConfig(granularity=granularity)
    g = construct_graph(pdb_path=str(HUMAN_PROTEOME / Path(file)))
    file_graph = os.path.splitext(file)[0].split("/")[0] + ".pkl"
    dump(g, open(str(HUMAN_PROTEOME_CA_GRAPHS / file_graph), "wb"))


def main():
    pdb_files = filter_pdb_files(os.listdir(HUMAN_PROTEOME))
    pdb_files = filter_monomers(
        [str(HUMAN_PROTEOME / Path(file)) for file in pdb_files]
    )
    make_dir(HUMAN_PROTEOME_CA_GRAPHS)

    Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(pdb_to_graph)(file) for file in pdb_files
    )


if __name__ == "__main__":
    main()
