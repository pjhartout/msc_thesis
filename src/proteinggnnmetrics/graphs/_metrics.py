# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices


def _parallel_pairwise(X1, X2, metric_params, n_jobs):
    parallel_kwargs = {"mmap_mode": "c"}
    # effective_metric_params = metric_params.copy()
    metric_func = np.linalg.norm
    n_columns = len(X2)
    distance_matrices = Parallel(n_jobs=n_jobs, **parallel_kwargs)(
        delayed(metric_func)(X1, X2[s])
        for s in gen_even_slices(n_columns, effective_n_jobs(n_jobs))
    )

    distance_matrices = np.stack(distance_matrices, axis=1)
    return distance_matrices
