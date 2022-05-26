#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""descriptor_debug_script.py

Scripts calling graph descriptors crash when computing graph statistics due to dependency problems. Here we want to figure out what is causing this.

"""

import random

import hydra
import numpy as np
import pandas as pd
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinmetrics.descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    LaplacianSpectrum,
)
from proteinmetrics.distance import MaximumMeanDiscrepancy
from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.kernels import LinearKernel
from proteinmetrics.loaders import list_pdb_files, load_descriptor, load_graphs
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates
from proteinmetrics.utils.functions import remove_fragments


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf")
def main(cfg: DictConfig) -> None:
    files = list_pdb_files(HUMAN_PROTEOME)
    files = remove_fragments(files)

    random_samples = random.Random(42).sample(files * 2, 10)
    midpoint = int(len(random_samples) / 2)

    graph_pipeline = pipeline.Pipeline(
        [
            ("coordinates", Coordinates(granularity="CA", n_jobs=4)),
            ("contact_graph", ContactMap(metric="euclidean", n_jobs=4)),
            ("eps_graph", EpsilonGraph(epsilon=8, n_jobs=4)),
            (
                "degree_histogram",
                DegreeHistogram(
                    graph_type="eps_graph", n_bins=10, n_jobs=4, verbose=True
                ),
            ),
            (
                "clustering_histogram",
                ClusteringHistogram(
                    graph_type="eps_graph",
                    n_bins=10,
                    density=True,
                    n_jobs=4,
                    verbose=True,
                ),
            ),
            (
                "laplacian_spectrum",
                LaplacianSpectrum(
                    graph_type="eps_graph", n_bins=10, n_jobs=4, verbose=True
                ),
            ),
        ]
    )

    proteins_1 = graph_pipeline.fit_transform(random_samples[midpoint:])
    proteins_2 = graph_pipeline.fit_transform(random_samples[:midpoint])

    descriptor_1 = load_descriptor(
        proteins_1,
        descriptor="laplacian_spectrum_histogram",
        graph_type="eps_graph",
    )
    descriptor_2 = load_descriptor(
        proteins_2,
        descriptor="laplacian_spectrum_histogram",
        graph_type="eps_graph",
    )
    mmd = MaximumMeanDiscrepancy(
        biased=True,
        kernel=LinearKernel(n_jobs=cfg.compute.n_jobs),  # type: ignore
    ).compute(descriptor_1, descriptor_2)
    print(mmd)


if __name__ == "__main__":
    main()
