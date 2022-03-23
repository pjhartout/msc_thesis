#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""filename.py

***file description***

"""

import os
import random

import numpy as np
import plotly.graph_objects as go
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline

from proteinggnnmetrics.descriptors import DegreeHistogram
from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph, KNNGraph
from proteinggnnmetrics.kernels import LinearKernel
from proteinggnnmetrics.loaders import (
    list_pdb_files,
    load_descriptor,
    load_graphs,
)
from proteinggnnmetrics.paths import CACHE_DIR, HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.perturbations import GaussianNoise
from proteinggnnmetrics.utils.functions import configure, flatten_lists

config = configure()

N_JOBS = int(config["COMPUTE"]["N_JOBS"])
REDUCE_DATA = config["DEBUG"]["REDUCE_DATA"]


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)

    if REDUCE_DATA:
        pdb_files = random.sample(pdb_files, 100)

    half = int(len(pdb_files) / 2)

    base_feature_steps = [
        ("coordinates", Coordinates(granularity="CA", n_jobs=N_JOBS)),
        ("contact map", ContactMap(metric="euclidean", n_jobs=N_JOBS)),
        ("epsilon graph", EpsilonGraph(epsilon=6.0, n_jobs=N_JOBS)),
        (
            "degree histogram",
            DegreeHistogram("eps_graph", n_bins=30, n_jobs=N_JOBS),
        ),
    ]

    base_feature_pipeline = pipeline.Pipeline(base_feature_steps, verbose=100)
    print("Building baseline graphs")
    proteins = base_feature_pipeline.fit_transform(pdb_files)

    mmds = list()
    params = list()
    for std in np.arange(0.1, 1.1, 0.1):
        print(f"Applying Gaussian noise with std={std}")
        perturb_feature_steps = flatten_lists(
            [
                base_feature_steps[:1]
                + [
                    (
                        "gauss",
                        GaussianNoise(
                            random_state=42,
                            noise_mean=0,
                            noise_variance=std,
                            n_jobs=N_JOBS,
                        ),
                    )
                ]
                + base_feature_steps[1:]
            ]
        )

        perturb_feature_pipeline = pipeline.Pipeline(
            perturb_feature_steps, verbose=100
        )
        proteins_perturbed = perturb_feature_pipeline.fit_transform(pdb_files)
        protein_graphs = load_descriptor(
            proteins, "degree_histogram", "eps_graph"
        )
        protein_graphs_perturbed = load_descriptor(
            proteins_perturbed, "degree_histogram", "eps_graph"
        )
        print("Calculating MMD")
        params.append({"std": std})
        mmds.append(
            MaximumMeanDiscrepancy(
                biased=True,
                squared=False,
                kernel=LinearKernel(dense_output=True,),
            ).compute(protein_graphs, protein_graphs_perturbed)
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[param["std"] for param in params],
            y=mmds,
            name="spline",
            line_shape="spline",
        )
    )
    fig.show()
    print("Done with perturbation")


if __name__ == "__main__":
    main()
