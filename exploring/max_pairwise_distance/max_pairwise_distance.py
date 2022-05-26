#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""max_pariwise_distance.py

What is the highest pairwise distance observed between any two residues in a protein?

Around 462.5783386230469

"""


import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gtda import pipeline

from proteinmetrics.graphs import ContactMap
from proteinmetrics.loaders import list_pdb_files
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates

N_JOBS = 40

log = logging.getLogger(__name__)


def main():
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(granularity="CA", n_jobs=N_JOBS, verbose=True),
        ),
        (
            "contact_map",
            ContactMap(n_jobs=N_JOBS, verbose=True),
        ),
    ]
    proteins = pipeline.Pipeline(base_feature_steps).fit_transform(pdb_files)
    max_distance = np.max(
        np.array([np.max(protein.contact_map) for protein in proteins])
    )
    print(f"Maximum distance: {max_distance}")
    log.info(f"Maximum distance: {max_distance}")


if __name__ == "__main__":
    main()
