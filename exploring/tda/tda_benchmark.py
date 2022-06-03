#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tda_giotto.py

Is using ripser directly using Bastian's data structure faster than giotto's?

"""
import os
import pickle
import random
from collections import namedtuple
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
from torch_topological.nn import VietorisRipsComplex
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
    distribute_function,
    flatten_lists,
    remove_fragments,
    tqdm_joblib,
)

N_JOBS = 4


@timeit
def giotto(proteins):
    tda_step = (
        TopologicalDescriptor(
            "diagram",
            epsilon=0.01,
            n_bins=100,
            order=2,
            n_jobs=6,
            landscape_layers=1,
            verbose=True,
            use_caching=False,
        ),
    )

    proteins = tda_step[0].fit_transform(proteins)
    return proteins


def compute_vr(coordinates, vr):
    return vr(coordinates)


@timeit
def torch_topological(proteins):
    # get coordinates
    coordinates = [protein.coordinates for protein in proteins]
    vr = VietorisRipsComplex(dim=3)
    vr_complexes = distribute_function(
        compute_vr,
        coordinates,
        n_jobs=N_JOBS,
        tqdm_label="Computing Vietoris-Rips complexes using torch-topological",
        show_tqdm=True,
        vr=vr,
    )


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    pdb_files = remove_fragments(pdb_files)
    sampled_files = random.Random(42).sample(pdb_files, 5 * 2)
    midpoint = int(len(sampled_files) / 2)
    base_feature_steps = [
        ("coordinates", Coordinates(granularity="CA", n_jobs=N_JOBS),),
        ("contact_map", ContactMap(n_jobs=N_JOBS, verbose=True,),),
    ]

    base_feature_pipeline = pipeline.Pipeline(base_feature_steps, verbose=True)
    proteins = base_feature_pipeline.fit_transform(sampled_files[:10],)

    giotto(proteins)

    torch_topological(proteins)


if __name__ == "__main__":
    main()
