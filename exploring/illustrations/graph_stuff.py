#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""illustrationss.py

Introduce gaussian noise to proteins and save resulting point cloud images.

"""

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as gobj
import seaborn as sns
from gtda import pipeline
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.loaders import list_pdb_files
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates, Sequence
from proteinmetrics.perturbations import GaussianNoise, Shear, Taper, Twist
from proteinmetrics.utils.debug import measure_memory, timeit
from proteinmetrics.utils.plots import setup_plotting_parameters

poi = 3


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    files = [
        file for file in pdb_files if "AF-A0A6Q8PFQ6-F1-model_v2" in str(file)
    ]
    setup_plotting_parameters()
    # base_feature_steps = [
    #     (
    #         "sequence",
    #         Sequence(n_jobs=6, verbose=True),
    #     ),
    # ]
    # base_feature_pipeline = pipeline.Pipeline(
    #     base_feature_steps, verbose=False
    # )
    # proteins = base_feature_pipeline.fit_transform(pdb_files)
    # sequences = [protein.sequence for protein in proteins]

    # graph = proteins[0].graphs["eps_graph"]
    # pos = nx.spring_layout(graph)
    # nx.draw(graph, pos=pos)

    # plt.savefig(
    #     here() / "exploring" / "illustrations" / "8_a_graph.pdf",
    #     bbox_inches="tight",
    # )

    base_feature_steps = [
        ("coordinates", Coordinates(granularity="CA", n_jobs=1),),
        ("contact_map", ContactMap(n_jobs=1,),),
        ("epsilon_graph", EpsilonGraph(epsilon=8, n_jobs=1,),),
    ]
    base_feature_pipeline = pipeline.Pipeline(
        base_feature_steps, verbose=False
    )
    proteins = base_feature_pipeline.fit_transform(files)
    graph = proteins[0].graphs["eps_graph"]
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos=pos)

    plt.savefig(
        here() / "exploring" / "illustrations" / "8_a_graph.pdf",
        bbox_inches="tight",
    )
    plt.clf()

    # total_edges is the total number of possible edges in a graph
    total_possible_edges = (
        graph.number_of_nodes() * (graph.number_of_nodes() - 1)
    ) / 2
    ba_graph = nx.erdos_renyi_graph(
        n=graph.number_of_nodes(),
        p=graph.number_of_edges() / total_possible_edges,
    )
    pos = nx.spring_layout(ba_graph)
    nx.draw(ba_graph, pos=pos)

    plt.savefig(
        here() / "exploring" / "illustrations" / "er_graph_8_a_params.pdf",
        bbox_inches="tight",
    )
    plt.clf()


if __name__ == "__main__":
    main()
