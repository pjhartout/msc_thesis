#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tda_kernel_debug.py

Here we want to debug the tda persistence fisher pipeline business

"""

import os
import pickle
import random
from operator import truediv

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from gtda.homology import VietorisRipsPersistence
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinmetrics.descriptors import LaplacianSpectrum, TopologicalDescriptor
from proteinmetrics.distance import (
    MaximumMeanDiscrepancy,
    PearsonCorrelation,
    SpearmanCorrelation,
)
from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.kernels import (
    LinearKernel,
    MultiScaleKernel,
    PersistenceFisherKernel,
)
from proteinmetrics.loaders import list_pdb_files, load_descriptor, load_graphs
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates
from proteinmetrics.perturbations import GaussianNoise, Twist
from proteinmetrics.utils.debug import SamplePoints, measure_memory, timeit
from proteinmetrics.utils.functions import (
    flatten_lists,
    remove_fragments,
    tqdm_joblib,
)

N_JOBS = 4


@timeit
def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    pdb_files = remove_fragments(pdb_files)
    sampled_files = random.Random(42).sample(pdb_files, 5)
    midpoint = int(len(sampled_files) / 2)
    base_feature_steps = [
        ("coordinates", Coordinates(granularity="CA", n_jobs=N_JOBS),),
        ("sample", SamplePoints(n=2)),
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
                n_jobs=6,
                landscape_layers=1,
                verbose=True,
                use_caching=True,
            ),
        ),
    ]

    base_feature_pipeline = pipeline.Pipeline(base_feature_steps, verbose=True)
    proteins = base_feature_pipeline.fit_transform(sampled_files[midpoint:],)

    results = list()
    for twist in tqdm(
        np.arange(0, 0.1, 0.05,), position=1, leave=False, desc="Twist range",
    ):
        perturb_feature_steps = flatten_lists(
            [
                base_feature_steps[:1]
                + [
                    (
                        "twist",
                        Twist(
                            alpha=twist,
                            random_state=42,
                            n_jobs=N_JOBS,
                            verbose=True,
                        ),
                    )
                ]
                + base_feature_steps[1:]
            ]
        )
        perturb_feature_pipeline = pipeline.Pipeline(perturb_feature_steps)
        proteins_perturbed = perturb_feature_pipeline.fit_transform(
            sampled_files[:midpoint]
        )

        diagrams = [
            protein.descriptors["contact_graph"]["diagram"]
            for protein in proteins
        ]
        diagrams_perturbed = [
            protein.descriptors["contact_graph"]["diagram"]
            for protein in proteins_perturbed
        ]

        # Computer kernel of single combo:
        kernel = MultiScaleKernel(sigma=1.0, n_jobs=1)
        res = kernel.compute_matrix(diagrams, diagrams_perturbed)

        mmd_tda = MaximumMeanDiscrepancy(
            biased=True,
            squared=True,
            kernel=MultiScaleKernel(sigma=1.0, n_jobs=N_JOBS),  # type: ignore
        ).compute(diagrams, diagrams_perturbed)

        results.append({"mmd_tda": mmd_tda, "twist": twist})

    print("Dumping results")
    results = pd.DataFrame(results).to_csv(
        here()
        / "data/experiments/tda_twist"
        / "mmd_single_run_twist_exploring.csv"
    )


if __name__ == "__main__":
    main()
