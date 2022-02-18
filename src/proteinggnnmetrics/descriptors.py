# -*- coding: utf-8 -*-

"""descriptors.py

Graph descriptors essentially extract fixed-length representation of a graph

TODO: check docstrings, citations
"""

from abc import ABCMeta
from typing import Any, List

import networkx as nx
import numpy as np

from .constants import N_JOBS
from .protein import Protein
from .utils.utils import distribute_function


class Descriptor(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self, n_jobs):
        self.n_jobs = n_jobs

    def describe(self):
        """Applies descriptor to graph
        """
        pass


class DegreeHistogram(Descriptor):
    def __init__(self, graph_type: str, hist_len: int, n_jobs: int = N_JOBS):
        self.hist_len = hist_len
        self.graph_type = graph_type
        self.n_jobs = n_jobs

    def describe(self, proteins: List[Protein]) -> Any:
        def calculate_degree_histogram(protein: Protein, normalize=True):
            G = protein.get_nx_graph(self.graph_type)
            degrees = np.array([val for (node, val) in G.degree()])
            histogram = np.bincount(degrees, minlength=self.hist_len + 1)
            if normalize:
                histogram = histogram / np.sum(histogram)
            return histogram

        histograms = distribute_function(
            calculate_degree_histogram,
            proteins,
            "Compute degree histogram",
            self.n_jobs,
        )
        return histograms
