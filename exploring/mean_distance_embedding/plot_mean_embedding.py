#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_mean_embedding.py

"""


import argparse
import json
import logging
import os
import random
import sys
from enum import unique
from multiprocessing.sharedctypes import Value
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here
from sklearn import cluster
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from proteinmetrics.descriptors import (
    ESM,
    ClusteringHistogram,
    DegreeHistogram,
    DistanceHistogram,
    LaplacianSpectrum,
    RamachandranAngles,
    TopologicalDescriptor,
)
from proteinmetrics.distance import MaximumMeanDiscrepancy
from proteinmetrics.graphs import ContactMap, EpsilonGraph, KNNGraph
from proteinmetrics.kernels import (
    LinearKernel,
    MultiScaleKernel,
    PersistenceFisherKernel,
)
from proteinmetrics.loaders import list_pdb_files, load_descriptor
from proteinmetrics.paths import DATA_HOME, ECOLI_PROTEOME, HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates, Sequence
from proteinmetrics.perturbations import (
    AddConnectedNodes,
    AddEdges,
    GaussianNoise,
    GraphPerturbation,
    Mutation,
    RemoveEdges,
    RewireEdges,
    Shear,
    Taper,
    Twist,
)
from proteinmetrics.protein import Protein
from proteinmetrics.utils.debug import SamplePoints
from proteinmetrics.utils.functions import (
    distribute_function,
    flatten_lists,
    make_dir,
    remove_fragments,
    save_obj,
)
from proteinmetrics.utils.plots import setup_plotting_parameters

log = logging.getLogger(__name__)


def load_data():
    clustering = pd.read_csv(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "mean_distance_embedding_clustering_histogram.csv",
        index_col=0,
    )
    clustering = clustering.assign(descriptor="clustering")
    clustering = clustering.assign(eps="8")

    clustering_high_eps = pd.read_csv(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "mean_distance_embedding_clustering_histogram_high_eps.csv",
        index_col=0,
    )
    clustering_high_eps = clustering_high_eps.assign(descriptor="clustering")
    clustering_high_eps = clustering_high_eps.assign(eps="32")

    degree = pd.read_csv(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "mean_distance_embedding_degree_histogram.csv",
        index_col=0,
    )
    degree = degree.assign(descriptor="degree")
    degree = degree.assign(eps="8")

    degree_high_eps = pd.read_csv(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "mean_distance_embedding_degree_histogram_high_eps.csv",
        index_col=0,
    )
    degree_high_eps = degree_high_eps.assign(descriptor="degree")
    degree_high_eps = degree_high_eps.assign(eps="32")

    laplacian = pd.read_csv(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "mean_distance_embedding_laplacian_spectrum.csv",
        index_col=0,
    )
    laplacian = laplacian.assign(descriptor="laplacian")
    laplacian = laplacian.assign(eps="8")

    laplacian_high_eps = pd.read_csv(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "mean_distance_embedding_laplacian_spectrum_high_eps.csv",
        index_col=0,
    )
    laplacian_high_eps = laplacian_high_eps.assign(descriptor="laplacian")
    laplacian_high_eps = laplacian_high_eps.assign(eps="32")

    dihedral = pd.read_csv(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "mean_distance_embedding_dihedral_hist.csv",
        index_col=0,
    )
    dihedral = dihedral.assign(descriptor="dihedral")

    distance = pd.read_csv(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "mean_distance_embedding_dist_hist.csv",
        index_col=0,
    )
    distance = distance.assign(descriptor="distance")

    df = pd.concat(
        [
            clustering,
            degree,
            laplacian,
            clustering_high_eps,
            degree_high_eps,
            laplacian_high_eps,
            dihedral,
            distance,
        ]
    )
    return df


def main():
    data = load_data()
    data = data.rename(columns={"dists": "Euclidean distance"})
    setup_plotting_parameters()
    protein_descriptor = data.loc[
        data.descriptor.isin(["dihedral", "distance"])
    ]
    graph_descriptor = data.loc[
        ~data.descriptor.isin(["dihedral", "distance"])
    ]

    palette = sns.color_palette("mako_r", graph_descriptor.eps.nunique())
    g = sns.FacetGrid(
        graph_descriptor, col="descriptor", col_wrap=3, sharey=False,
    )
    g.map_dataframe(
        sns.violinplot,
        x="descriptor",
        y="Euclidean distance",
        hue="eps",
        palette=palette,
        split=True,
    )
    g.set(xticklabels=[])
    g.set(xlabel=None)

    g.add_legend(title=r"$\varepsilon$-value")

    # sns.violinplot(
    #     col="descriptor",
    #     y="dists",
    #     data=data,
    #     facet_kws={"sharey": False, "sharex": True},
    #     split=True,
    # )

    titles = [
        "Clustering Coefficient Histogram",
        "Degree Distribution Histogram",
        "Laplacian Spectrum Histogram",
    ]
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "violin_graph_descriptors.pdf",
    )

    #### Protein descriptors

    palette = sns.color_palette(
        "mako_r", protein_descriptor.descriptor.nunique()
    )
    g = sns.FacetGrid(
        protein_descriptor, col="descriptor", col_wrap=2, sharey=False,
    )
    g.map_dataframe(
        sns.violinplot,
        x="descriptor",
        y="Euclidean distance",
        # hue="eps",
        palette=palette,
        split=True,
    )
    g.set(xticklabels=[])
    g.set(xlabel=None)

    titles = [
        "Dihedral Angles Histogram",
        "Pairwise Interatomic Distance Histogram",
    ]
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(
        here()
        / "exploring"
        / "mean_distance_embedding"
        / "violin_protein_descriptors.pdf",
    )


if __name__ == "__main__":
    main()
