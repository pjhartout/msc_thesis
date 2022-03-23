# -*- coding: utf-8 -*-

"""descriptors.py

Graph descriptors essentially extract fixed-length representation of a graph

"""

from abc import ABCMeta
from typing import Any, Iterable

import numpy as np

from .protein import Protein


class Perturbations(metaclass=ABCMeta):
    """Defines skeleton of perturbation classes classes"""

    def __init__(self, random_state):
        self.random_state = random_state

    def fit_transform(self, X: Any) -> Any:
        """Apply perturbation to graph or graph representation"""
        return X


class GaussianNoise(Perturbations):
    """Adds Gaussian noise to coordinates"""

    def __init__(
        self, random_state: int, noise_mean: float, noise_variance: float
    ) -> None:
        self.random_state = random_state
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance

    def add_noise_to_protein(self, protein: Protein) -> Protein:
        noise = np.random.normal(0, 1, len(protein.coordinates))
        protein.coordinates

    def fit_transform(self, X: Iterable[Protein]) -> Any:
        return X
