#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""descriptor_range.py

What is:
    - The range of the laplacian
    - The range of the degree?

max degree 719
laplacian  1.2586714829332633
distance   462.5783386230469
"""
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from gtda import pipeline

from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.loaders import list_pdb_files
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates, Sequence

N_JOBS = 4

log = logging.getLogger(__name__)


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)

    base_feature_steps = [
        ("sequence", Sequence(n_jobs=N_JOBS, verbose=True),),
    ]

    proteins = pipeline.Pipeline(base_feature_steps).fit_transform(pdb_files)
    max_length = max([len(protein.sequence) for protein in proteins])
    print(max_length)


if __name__ == "__main__":
    main()
