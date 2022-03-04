# -*- coding: utf-8 -*-

"""descriptors.py

Graph descriptors essentially extract fixed-length representation of a graph

"""

from abc import ABCMeta
from typing import Any

import numpy as np


class Perturbations(metaclass=ABCMeta):
    """Defines skeleton of perturbation classes classes"""

    def __init__(self, random_state):
        self.random_state = random_state
        pass

    def fit_transform(self, X: Any) -> np.ndarray:
        """Apply perturbation to graph or graph representation
        """
        pass
