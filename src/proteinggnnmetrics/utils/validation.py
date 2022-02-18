# -*- coding: utf-8 -*-

"""validation.py

Performs basic validation checks of data objects

"""

import os
from pathlib import Path, PosixPath
from typing import Any, List

import numpy as np

from ..errors import FileExtentionError


def check_graphs(X, distance_matrices=False, **kwargs):
    kwargs_ = {"force_all_finite": not distance_matrices}
    kwargs_.update(kwargs)
    kwargs_.pop("allow_nd", None)
    kwargs_.pop("ensure_2d", None)
    if hasattr(X, "shape") and hasattr(X, "ndim"):
        if X.ndim != 3:
            if X.ndim == 2:
                extra_2D = (
                    "\nReshape your input X using X.reshape(1, *X.shape) or "
                    "X[None, :, :] if X is a single point cloud/distance "
                    "matrix/adjacency matrix of a weighted graph."
                )
            else:
                extra_2D = ""
            raise ValueError(
                f"Input must be a single 3D array or a list of 2D arrays or "
                f"sparse matrices. Structure of dimension {X.ndim} passed."
                + extra_2D
            )
    return X


def check_fnames(files: List[PosixPath]) -> List[PosixPath]:
    if type(files) == list:
        pathname_list = [Path(file) for file in files]
        for file in pathname_list:
            if file.suffix != ".pdb":
                raise FileExtentionError(
                    "Make sure you're only processing PDB files!"
                )
        return pathname_list
    return files


def check_dist(dist: Any) -> Any:
    """TODO: implement distribution checks"""
    dist = np.asarray(dist)
    return dist


if __name__ == "__main__":
    main()
