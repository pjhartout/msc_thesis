# -*- coding: utf-8 -*-

"""descriptors.py

Graph descriptors essentially extract fixed-length representation of a graph

"""

from abc import ABCMeta
from typing import Any, Iterable, List

import numpy as np

from .protein import Protein
from .utils.functions import distribute_function


class Perturbations(metaclass=ABCMeta):
    """Defines skeleton of perturbation classes classes"""

    def __init__(self, random_state):
        self.random_state = random_state

    def fit(self, proteins: List[Protein]) -> List[Protein]:
        """For pipeline compatibiltiy"""
        pass

    def transform(self, X: List[Protein]) -> None:
        return X

    def fit_transform(self, X: Any) -> Any:
        """Apply perturbation to graph or graph representation"""
        return X


class GaussianNoise(Perturbations):
    """Adds Gaussian noise to coordinates"""

    def __init__(
        self,
        random_state: int,
        noise_mean: float,
        noise_variance: float,
        n_jobs: int,
        verbose: bool = False,
    ) -> None:
        self.random_state = random_state
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        self.n_jobs = n_jobs
        self.verbose = verbose

    def add_noise_to_protein(self, protein: Protein) -> Protein:
        noise = np.random.normal(0, 1, len(protein.coordinates) * 3).reshape(
            len(protein.coordinates), 3
        )
        protein.coordinates = protein.coordinates + noise
        return protein

    def fit(self, X: List[Protein]) -> None:
        pass

    def transform(self, X: List[Protein]) -> None:
        return X

    def fit_transform(self, X: Iterable[Protein], y=None) -> Any:
        X = distribute_function(
            self.add_noise_to_protein,
            X,
            self.n_jobs,
            "Adding Gaussian noise to proteins",
            show_tqdm=self.verbose,
        )
        return X
