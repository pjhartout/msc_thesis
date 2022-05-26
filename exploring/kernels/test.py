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
from sklearn.feature_extraction import DictVectorizer

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


def matrix2df(matrix, column_labels):
    return pd.DataFrame(matrix, columns=column_labels).fillna(0)


def fill_missing_cols(X, Y):
    mismatched_cols = set(X.columns) ^ set(Y.columns)
    for col in mismatched_cols:
        if col in X:
            Y[col] = 0
        else:
            X[col] = 0
    return X, Y


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

    vectorizer = DictVectorizer(dtype=np.uint8, sparse=False)
    X = vectorizer.fit_transform(X)
    column_labels = vectorizer.get_feature_names_out()
    X = matrix2df(X, column_labels)

    if Y == None:
        Y = X
    else:
        Y = vectorizer.fit_transform(Y)
        column_labels = vectorizer.get_feature_names_out()
        Y = matrix2df(Y, column_labels)

    X, Y = fill_missing_cols(X, Y)
    X, Y = X.values, Y.values
    K = X.dot(Y.T)

    if X.equals(Y):
        # Make matrix symmetric
        i_upper = np.triu_indices(n, 1)
        i_lower = np.tril_indices(n, -1)
        K[i_lower] = K[i_upper]
    print(K)


if __name__ == "__main__":
    main()
