# -*- coding: utf-8 -*-
"""benchmarks.py

Here we make benchmarks for the time it takes to run the various
parts of the MMD pipeline configs.

"""


import time

import hydra
import pandas as pd
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf.dictconfig import DictConfig
from pyprojroot import here

from proteinmetrics.descriptors import (
    ESM,
    ClusteringHistogram,
    DegreeHistogram,
    DistanceHistogram,
    LaplacianSpectrum,
    RamachandranAngles,
    TopologicalDescriptor,
)
from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.kernels import (
    GaussianKernel,
    LinearKernel,
    PersistenceFisherKernel,
)
from proteinmetrics.loaders import list_pdb_files, load_descriptor, load_graphs
from proteinmetrics.paths import DATA_HOME, HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates, Sequence
from proteinmetrics.utils.debug import measure_memory, timeit
from proteinmetrics.utils.functions import make_dir

N_JOBS = 10
N_SAMPLES = 100


@timeit
def distance_benchmark(pdb_files):
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=N_JOBS),
        ),
        (
            "contact map",
            ContactMap(
                metric="euclidean",
                n_jobs=N_JOBS,
            ),
        ),
        (
            "epsilon graph",
            DistanceHistogram(
                bin_range=(0, 1000), n_bins=1000, n_jobs=N_JOBS, verbose=True
            ),
        ),
    ]
    start = time.perf_counter()
    proteins = pipeline.Pipeline(
        base_feature_steps, verbose=True
    ).fit_transform(pdb_files)
    time_elapsed = time.perf_counter() - start
    return time_elapsed


@timeit
def angles_benchmark(pdb_files):
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="backbone", n_jobs=N_JOBS),
        ),
        (
            "epsilon graph",
            RamachandranAngles(
                from_pdb=False, n_bins=1000, n_jobs=N_JOBS, verbose=True
            ),
        ),
    ]
    start = time.perf_counter()
    proteins = pipeline.Pipeline(
        base_feature_steps, verbose=True
    ).fit_transform(pdb_files)
    time_elapsed = time.perf_counter() - start
    return time_elapsed


def reps_benchmarks(pdb_files):
    distance_histogram_elapsed = distance_benchmark(pdb_files)
    angles_benchmark_elapsed = angles_benchmark(pdb_files)
    return distance_histogram_elapsed, angles_benchmark_elapsed


@hydra.main(
    version_base=None, config_path=str(here()) + "/conf", config_name="conf"
)
def main(cfg: DictConfig):
    # Potential performance improvements in the real world:
    # - not a representative RAM footprint
    # - not a fully representative scaling/sample size
    print(f"Looking for data in {HUMAN_PROTEOME}")
    print(f"Test existence of {len(list_pdb_files(HUMAN_PROTEOME))} files")
    make_dir(here() / cfg.meta.time_estimates_dir)

    # Reprensentations benchmarks
    distance_histogram_elapsed, angles_benchmark_elapsed = reps_benchmarks(
        list_pdb_files(HUMAN_PROTEOME)[:N_SAMPLES]
    )

    representation_benchmarks = pd.DataFrame(
        data=[distance_histogram_elapsed, angles_benchmark_elapsed],
        index=["distance_histogram", "angles_histogram"],
        columns=["representation"],
    )

    representation_benchmarks.to_csv(
        here() / cfg.meta.time_estimates_dir / "protein_benchmarks.csv"
    )


if __name__ == "__main__":
    main()
