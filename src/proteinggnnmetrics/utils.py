# -*- coding: utf-8 -*-
"""utils.py

Provides various utilities useful for the project
"""

import contextlib
import os
from typing import List

import joblib
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
