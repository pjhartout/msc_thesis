#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tda_features_distance.py

Playing around with distances computed from TDA features

"""


from struct import pack

import matplotlib.pyplot as plt
import numpy as np

from proteinmetrics.distance import (
    MinkowskyDistance,
    TopologicalPairwiseDistance,
)
from proteinmetrics.loaders import load_descriptor, load_proteins
from proteinmetrics.paths import CACHE_DIR
from proteinmetrics.utils.debug import measure_memory, timeit
from proteinmetrics.utils.functions import distribute_function


def main():
    proteins = load_proteins(CACHE_DIR / "sample_human_proteome_alpha_fold")
    half = int(len(proteins) / 2)
    diagrams = load_descriptor(proteins, "diagram", "contact_graph")
    wasserstein_distance = TopologicalPairwiseDistance(
        metric="wasserstein",
        metric_params={"p": 2},
        order=2,
        n_jobs=int(config["COMPUTE"]["N_JOBS"]),
    )

    pairwise_distances = wasserstein_distance.fit_transform(diagrams)

    # show heatmap of pairwise distances

    minkowsky_distance = MinkowskyDistance(p=2)
    distance = minkowsky_distance.fit_transform(
        pairwise_distances, pairwise_distances
    )
    print(distance)


if __name__ == "__main__":
    main()
