#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""timings.py

Here we want to get an accurate time estimate of the computation time for some tasks.

"""

import hydra
from gtda import pipeline
from omegaconf.dictconfig import DictConfig
from pyprojroot import here

from proteinggnnmetrics.descriptors import TopologicalDescriptor
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.utils.debug import measure_memory, timeit

N_JOBS = 4


@timeit
def tda_benchmark(pdb_files):
    base_feature_steps = [
        ("coordinates", Coordinates(granularity="CA", n_jobs=N_JOBS),),
        ("contact_map", ContactMap(n_jobs=N_JOBS, verbose=True,),),
        (
            "epsilon_graph",
            EpsilonGraph(n_jobs=N_JOBS, epsilon=8, verbose=True,),
        ),
        (
            "tda",
            TopologicalDescriptor(
                "diagram",
                epsilon=0.01,
                n_bins=100,
                order=2,
                n_jobs=N_JOBS,
                landscape_layers=1,
                verbose=True,
            ),
        ),
    ]

    proteins = pipeline.Pipeline(
        base_feature_steps, verbose=True
    ).fit_transform(pdb_files)
    return proteins


@timeit
def esm_benchmark(pdb_files):
    pass


@timeit
def graphs_benchmark(pdb_files):
    pass


def reps_benchmarks(pdb_files):
    proteins = tda_benchmark(pdb_files)
    proteins = esm_benchmark(pdb_files)
    proteins = graphs_benchmark(pdb_files)


@hydra.main(config_path=str(here()) + "/conf", config_name="conf")
def main(cfg: DictConfig):

    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    pdb_files = pdb_files[60:]

    proteins = reps_benchmarks(pdb_files)


if __name__ == "__main__":
    main()
