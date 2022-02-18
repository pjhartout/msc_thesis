# -*- coding: utf-8 -*-

"""descriptors.py

Graph descriptors essentially extract fixed-length representation of a graph

"""

from abc import ABCMeta
from typing import Any

import numpy as np


class Descriptor(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self):
        pass

    @staticmethod
    def describe(X: Any) -> Any:
        """Applies descriptor to graph
        """
        pass


class DegreeDistributionHistogram(Descriptor):
    def __init__(self):
        pass

    def describe(G):
        degrees = [G.degree(n) for n in G.nodes()]
