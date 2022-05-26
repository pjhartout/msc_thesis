#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""filename.py

***file description***

"""

from collections import Counter
from itertools import combinations, combinations_with_replacement, product
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from proteinmetrics.utils.functions import chunks, flatten_lists


def get_hash(G):
    return dict(
        Counter(
            flatten_lists(
                list(
                    nx.weisfeiler_lehman_subgraph_hashes(
                        G, iterations=1,
                    ).values()
                )
            )
        )
    )


def dstack_product(x, y):
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)


def organize_into_matrix(lst, m, n):
    cnt = 0
    for idx, elem in enumerate(lst):
        cnt += 1
        if idx % m == 0:
            cnt = 0
        print(cnt)


def main():
    g_1 = nx.cubical_graph()
    g_2 = nx.cubical_graph()
    g_3 = nx.cubical_graph()

    hashes_1 = get_hash(g_1)

    g_2.remove_edge(1, 2)

    hashes_2 = get_hash(g_2)

    g_3.remove_edge(1, 2)
    g_3.remove_edge(3, 2)

    hashes_3 = get_hash(g_3)

    hashes_4 = get_hash(g_1)

    g_2.remove_edge(3, 5)

    hashes_5 = get_hash(g_2)

    g_3.remove_edge(4, 5)

    hashes_6 = get_hash(g_3)

    g_3 = nx.cubical_graph()

    hashes_7 = get_hash(g_3)

    X = [hashes_1, hashes_2, hashes_3]
    Y = [hashes_4, hashes_4, hashes_6, hashes_7]

    X = (
        pd.DataFrame(X)
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "sample_index"})
    )
    Y = (
        pd.DataFrame(Y)
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "sample_index"})
    )

    full = pd.concat([X, Y], ignore_index=True, sort=False)
    m, n = len(X), len(Y)
    res = np.zeros((m, n))
    for combi in combinations(full.index, 2):
        x = int(full.loc[combi[0]]["sample_index"])
        y = int(full.loc[combi[1]]["sample_index"])
        res[x][y] = (
            full.loc[combi[0]].multiply(full.loc[combi[1]], fill_value=0).sum()
        )
    print(res)

    # # Now the optimized KXX Gram matrix
    # full = pd.concat([X, X], ignore_index=True, sort=False).drop_duplicates()
    # n = len(X)
    # res = np.zeros((n, n))
    # # Just combinations if diag is not to be computed.
    # for combi in combinations_with_replacement(full.index, 2):
    #     x = int(full.loc[combi[0]]["sample_index"])
    #     y = int(full.loc[combi[1]]["sample_index"])
    #     res[x][y] = (
    #         full.loc[combi[0]].multiply(full.loc[combi[1]], fill_value=0).sum()
    #     )

    # i_upper = np.triu_indices(n, 1)
    # i_lower = np.tril_indices(n, -1)
    # res[i_lower] = res[i_upper]
    # print(res)


if __name__ == "__main__":
    main()
