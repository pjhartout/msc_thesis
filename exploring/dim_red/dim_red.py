#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""dim_red.py

The idea here is to look at the various representations of a protein and subsequently look at whether or not clusters emerge which we could simulate as modes to tweak the distribution of said modes.

"""
import os
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pyprojroot import here
from sklearn.decomposition import PCA
from torch import embedding
from tqdm import tqdm

from proteinggnnmetrics.descriptors import ESM
from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.kernels import LinearKernel
from proteinggnnmetrics.loaders import list_pdb_files, load_graphs
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates, Sequence
from proteinggnnmetrics.perturbations import Mutation
from proteinggnnmetrics.utils.functions import (
    flatten_lists,
    load_obj,
    remove_fragments,
    tqdm_joblib,
)
from proteinggnnmetrics.utils.plots import (
    setup_annotations,
    setup_plotting_parameters,
)

setup_plotting_parameters()


@hydra.main(
    version_base=None,
    config_path=str(here()) + "/conf/",
    config_name="dim_red",
)
def main(cfg):
    proteins = load_obj(here() / "constant_reps.pkl")
    # Phi/psi angles
    angles = np.array([protein.phi_psi_angles for protein in proteins])
    mapper = umap.UMAP()
    map = mapper.fit_transform(angles)
    plt.scatter(map[:, 0], map[:, 1])
    plt.savefig("./angles_umap.png")

    distance_hist = np.array([protein.distance_hist for protein in proteins])
    mapper = umap.UMAP()
    map = mapper.fit_transform(distance_hist)
    plt.scatter(map[:, 0], map[:, 1])
    plt.savefig("./distance_hist_umap.png")


if __name__ == "__main__":
    main()
