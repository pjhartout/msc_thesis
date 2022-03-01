# -*- coding: utf-8 -*-
"""utils.py

Provides various utilities useful for the project
"""

import contextlib
import os
from typing import Any, Callable, Iterable, List

import joblib
import networkx as nx
import numpy as np
from grakel import Graph, graph_from_networkx
from joblib import Parallel, delayed
from tqdm import tqdm


def filter_pdb_files(lst_of_files: List[str]) -> List[str]:
    """Filters pdb files from directory listing

    Args:
        lst_of_files (List[str]): list containing filenames, possibly including pdb
        files

    Returns:
        List: str
    """
    return [file for file in lst_of_files if file.endswith("pdb")]


def filter_monomers(lst_of_files: List[str]) -> List[str]:
    """Filters pdb files from directory listing

    Args:
        lst_of_files (List[str]): list containing filenames, possibly including
        monomers

    Returns:
        List: str
    """
    monomers = []
    for file in tqdm(lst_of_files):
        with open(file) as f:
            if "MONOMER" in f.read():
                monomers.append(file)
    return monomers


def make_dir(directory: str) -> None:
    """Idempotent function making directory and does not stop if it is already
    created.

    Args:
        directory (str): directory to be created
    """

    try:
        os.mkdir(directory)
    except OSError:
        print(
            f"Creation of the directory {directory} failed, probably already "
            f"exists"
        )
    else:
        print(f"Successfully created the directory {directory}")


def write_matrix(matrix, fname):
    """TODO: docstring"""
    with open(fname, "wb") as f:
        np.save(f, matrix)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar.

    Code stolen from https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def distribute_function(
    func: Callable, X: Any, tqdm_label: str, n_jobs: int
) -> Any:
    """Simply distributues the execution of func across multiple cores to process X faster"""
    with tqdm_joblib(tqdm(desc=tqdm_label, total=len(X),)) as progressbar:
        Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x) for x in X)
    return Xt


def networkx2grakel(X: Iterable) -> Iterable:
    Xt = list(graph_from_networkx(X, node_labels_tag="residue"))
    return Xt
