# -*- coding: utf-8 -*-
"""utils.py

Provides various utilities useful for the project
"""

import functools
import os
import time
from typing import List

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


def timeit(func):
    """timeit's doc"""

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"Function: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure
