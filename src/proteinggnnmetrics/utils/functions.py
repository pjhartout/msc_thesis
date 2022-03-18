# -*- coding: utf-8 -*-
"""utils.py

Provides various utilities useful for the project
"""

import configparser
import contextlib
import os
from itertools import product
from random import choice
from string import ascii_letters
from typing import Any, Callable, Dict, Iterable, List

import joblib
import networkx as nx
import numpy as np
from grakel import graph_from_networkx
from joblib import Parallel, delayed
from matplotlib.path import Path
from networkx.readwrite.graph6 import n_to_data
from pyprojroot import here
from tqdm import tqdm

from .exception import UniquenessError


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
    func: Callable,
    X: Iterable,
    n_jobs: int,
    tqdm_label: str = None,
    total: int = 1,
    show_tqdm: bool = True,
    **kwargs,
) -> Any:
    """Simply distributes the execution of func across multiple cores to process X faster"""
    if total == 1:
        total = len(X)

    if show_tqdm:
        with tqdm_joblib(tqdm(desc=tqdm_label, total=total)) as progressbar:
            Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x, **kwargs) for x in X)
    else:
        Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x, **kwargs) for x in X)
    return Xt


def networkx2grakel(X: Iterable) -> Iterable:
    Xt = list(graph_from_networkx(X, node_labels_tag="residue"))
    return Xt


def flatten_lists(lists: list) -> list:
    """Removes nested lists"""
    result = list()
    for _list in lists:
        _list = list(_list)
        if _list != []:
            result += _list
        else:
            continue
    return result


def configure() -> Dict:
    config = configparser.ConfigParser()
    config.read(here() / "setting.conf")
    return config


def chunks(lst, n):
    """returns lst divided into n chunks approx. the same size"""
    k, m = divmod(len(lst), n)
    return (
        lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    )


def generate_random_strings(string_length: int, n_strings: int) -> List[str]:
    unique_strings = list()
    for idx, item in enumerate(product(ascii_letters, repeat=string_length)):
        if idx == n_strings:
            break
        unique_strings.append("".join(item))

    if len(unique_strings) != n_strings:
        raise UniquenessError(
            f"Cannot generate enough unique strings from given string length "
            "of {string_length}. Please increase the string_length to continue."
        )
    return unique_strings
