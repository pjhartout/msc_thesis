#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""filename.py

***file description***

"""

from collections import Counter, defaultdict
from hashlib import blake2b
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx

from proteinggnnmetrics.utils.functions import flatten_lists


def dot_product(dicts: Tuple) -> int:
    # 0 * x = 0 so we only need to iterate over common keys
    return sum(dicts[0][key] * dicts[1].get(key, 0) for key in dicts[0])


def main():
    g_1 = nx.cubical_graph()
    g_2 = nx.cubical_graph()

    hashes_1 = dict(
        Counter(
            flatten_lists(
                list(
                    nx.weisfeiler_lehman_subgraph_hashes(
                        g_1, iterations=1,
                    ).values()
                )
            )
        )
    )

    print("--------")
    print("SECOND")
    print("--------")

    hashes_2 = dict(
        Counter(
            flatten_lists(
                list(
                    nx.weisfeiler_lehman_subgraph_hashes(
                        g_2, iterations=1,
                    ).values()
                )
            )
        )
    )

    print(hashes_1)
    print(hashes_2)
    print(dot_product((hashes_1, hashes_2)))


if __name__ == "__main__":
    main()
