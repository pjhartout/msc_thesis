# -*- coding: utf-8 -*-

"""validation.py

Performs basic validation checks of data objects

"""

import os


def _check_array_mod(X, **kwargs):
    """Modified version of :func:`sklearn.utils.validation.check_array. When
    keyword parameter `force_all_finite` is set to False, NaNs are not
    accepted but infinity is."""
    if not kwargs.get("force_all_finite", True):
        Xnew = check_array(X, **kwargs)
        if np.isnan(Xnew if not issparse(Xnew) else Xnew.data).any():
            raise ValueError(
                "Input contains NaNs. Only finite values and "
                "infinity are allowed when parameter "
                "`force_all_finite` is False."
            )
        return Xnew
    return check_array(X, **kwargs)


def check_graph(X, distance_matrices=False, **kwargs):
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


if __name__ == "__main__":
    main()
