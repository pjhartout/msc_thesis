# -*- coding: utf-8 -*-
"""utils.py

Provides various utilities useful for the project
"""

import configparser
import contextlib
import os
import pickle
from itertools import product
from pathlib import PosixPath
from random import choice
from string import ascii_letters
from typing import Any, Callable, Dict, Iterable, List, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from grakel import graph_from_networkx
from joblib import Parallel, delayed
from matplotlib.path import Path
from networkx.readwrite.graph6 import n_to_data
from pyprojroot import here
from tqdm import tqdm

from .exception import UniquenessError


def remove_fragments(files: List[PosixPath]) -> List[PosixPath]:
    """Some proteins are too long for AlphaFold to process, so it breaks it up into overlapping fragments. This can introduce bias in our data, so we only keep one fragment per protein.

    Args:
        files (List[PosixPath]): list of files to filter

    Returns:
        List[PosixPath]: list of filtered files
    """
    return [file for file in files if "F1" in str(file)]


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
    tqdm_label: str = "",
    total: int = 1,
    show_tqdm: bool = True,
    **kwargs,
) -> Any:
    """Simply distributes the execution of func across multiple cores to process X faster"""
    if total == 1:
        total = len(X)  # type: ignore

    if show_tqdm:
        with tqdm_joblib(tqdm(desc=tqdm_label, total=total)) as progressbar:
            Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x, **kwargs) for x in X)
    else:
        Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x, **kwargs) for x in X)
    return Xt


def networkx2grakel(X: Iterable) -> Iterable:
    Xt = list(graph_from_networkx(X, node_labels_tag="residue"))
    return Xt


def flatten_lists(lists: List) -> List:
    """Removes nested lists"""
    result = list()
    for _list in lists:
        _list = list(_list)
        if _list != []:
            result += _list
        else:
            continue
    return result


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


def pad_diagrams(Xt, homology_dimensions: Tuple = (0, 1)) -> np.ndarray:
    """Pads the diagrams to the same size."""
    # def filter(row,dim):
    #     return row[:2] == dim
    X_left = Xt[0]
    X_right = Xt[1]

    dim_shape_left = list()
    for dim in homology_dimensions:
        dim_shape_left.append(X_left[X_left[:, 2] == dim].shape[0])

    dim_shape_right = list()
    for dim in homology_dimensions:
        dim_shape_right.append(X_right[X_right[:, 2] == dim].shape[0])

    X_left = pd.DataFrame(X_left, columns=["birth", "death", "dim"])
    X_right = pd.DataFrame(X_right, columns=["birth", "death", "dim"])

    # Get unique dimensions
    max_dims = {
        dim: max([shape1, shape2])
        for dim, shape1, shape2 in zip(
            X_left["dim"].unique(), dim_shape_left, dim_shape_right
        )
    }

    # Find smallest value in each dimension for both dataframes
    min_birth_left = X_left.groupby("dim").min()["birth"].to_dict()
    min_birth_right = X_right.groupby("dim").min()["birth"].to_dict()
    min_values = {
        k: min(v, min_birth_right[k]) for k, v in min_birth_left.items()
    }
    for dim in max_dims.keys():
        # add n rows to dataframe
        X_left = pd.concat(
            [
                X_left,
                pd.DataFrame(
                    {
                        "birth": min_values[dim]
                        * (
                            max_dims[dim]
                            - len(X_left.loc[X_left["dim"] == dim])
                        ),
                        "death": min_values[dim]
                        * (
                            max_dims[dim]
                            - len(X_left.loc[X_left["dim"] == dim])
                        ),
                        "dim": [dim]
                        * (
                            max_dims[dim]
                            - len(X_left.loc[X_left["dim"] == dim])
                        ),
                    }
                ),
            ]
        )
        X_right = pd.concat(
            [
                X_right,
                pd.DataFrame(
                    {
                        "birth": min_values[dim]
                        * (
                            max_dims[dim]
                            - len(X_right.loc[X_right["dim"] == dim])
                        ),
                        "death": min_values[dim]
                        * (
                            max_dims[dim]
                            - len(X_right.loc[X_right["dim"] == dim])
                        ),
                        "dim": [dim]
                        * (
                            max_dims[dim]
                            - len(X_right.loc[X_right["dim"] == dim])
                        ),
                    }
                ),
            ]
        )

    Xt = np.array([X_left, X_right])
    return Xt  # typing: ignore


def positive_eig(K):
    """Assert true if the calculated kernel matrix is valid."""
    min_eig = np.real(np.min(np.linalg.eig(K)[0]))
    return min_eig


def distance2similarity(K):
    """
    Convert distance matrix to similarity matrix using a strictly
    monotone decreasing function.
    """
    K = np.exp(-K)
    return K


def save_obj(path: PosixPath, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_obj(path: PosixPath) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj
