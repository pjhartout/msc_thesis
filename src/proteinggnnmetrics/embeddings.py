# -*- coding: utf-8 -*-

"""embeddings.py

Classes to enable computing an embedding from a sequence given a model

TODO: check docstrings, citations
"""
import random
from abc import ABCMeta
from argparse import ArgumentError
from collections import Counter
from curses.ascii import EM
from faulthandler import disable
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
from joblib import Parallel, delayed
from pyprojroot import here
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    kneighbors_graph,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from tqdm import tqdm

from .protein import Protein
from .utils.functions import distribute_function, tqdm_joblib
from .utils.validation import check_graphs


class Embedding(metaclass=ABCMeta):
    def __init__(self, n_jobs, verbose) -> None:
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, sequences: List[Protein]) -> None:
        """Fit the embedding to the given sequences.

        Args:
            sequences (List[Protein]): list of sequences to embed.
        """
        pass

    def transform(self, proteins: List[Protein]) -> List[Protein]:
        """Transform the given sequences to embeddings.

        Args:
            sequences (List[Protein]): list of sequences to embed.

        Returns:
            List[Protein]: list of embeddings for each sequence.
        """
        return proteins

    def fit_transform(self, proteins: List[Protein]) -> List[Protein]:
        """Fit the embedding to the given sequences and transform them.

        Args:
            sequences (List[Protein]): list of sequences to embed.

        Returns:
            List[Protein]: list of embeddings for each sequence.
        """
        return proteins


class ESM(Embedding):
    _size_options = ["M", "XL"]

    def __init__(self, size: str, n_jobs: int, verbose: bool) -> None:
        super().__init__(n_jobs, verbose)
        self.size = size

    def fit(self, sequences: List[Protein], y=None) -> None:
        """Fit the embedding to the given sequences.

        Args:
            sequences (List[Protein]): list of sequences to embed.
        """
        pass

    def transform(self, sequences: List[Protein], y=None) -> List[Protein]:
        """Transform the given sequences to embeddings.

        Args:
            a (List[Protein]): list of sequences to embed.

        Returns:
            List[Protein]: list of embeddings for each sequence.
        """
        return sequences

    def fit_transform(self, proteins: List[Protein], y=None) -> List[Protein]:
        """Actual function used to compute embeddings of a list of
        proteins.
        The model is not fitted in this function but this is to ensure
        consistency with the rest of the library

        Args:
            sequences (List[Protein]): _description_

        Returns:
            List[Protein]: _description_
        """
        if self.verbose:
            print("Computing embeddings with ESM...")
            print("Loading model...")
        if self.size == "M":
            model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
            repr_layer = 6
        elif self.size == "XL":
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            repr_layer = 33
        else:
            raise ArgumentError(
                message=f"Size must be one of {self._size_options}",
                argument=None,
            )
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        # TODO: See if distribution makes sense here.
        if self.verbose:
            print("Getting sequences...")
        sequences = [
            (protein.name, protein.sequence)
            for protein in tqdm(proteins, disable=not self.verbose)
        ]
        _, _, batch_tokens = batch_converter(sequences)
        if self.verbose:
            print("Computing embeddings...")
        with torch.no_grad():
            results = model(
                batch_tokens, repr_layers=[repr_layer], return_contacts=True
            )
        token_representations = results["representations"][repr_layer]
        if self.verbose:
            print("Post-processing embeddings...")
        for idx, protein in tqdm(
            enumerate(proteins), total=len(proteins), disable=not self.verbose
        ):
            protein.embeddings["esm"] = (
                token_representations[idx].numpy().flatten()
            )
            print(protein.embeddings["esm"].shape)
        return proteins
