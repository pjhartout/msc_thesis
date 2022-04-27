#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""exp_3.py

The goal of this experiment is to investigate the effect of mutation on MMD features derived from ESM embeddings.

"""

import os
import random

import hydra
import numpy as np
import pandas as pd
from fastwlk import WeisfeilerLehmanKernel
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here
from torch import embedding
from tqdm import tqdm

from proteinggnnmetrics.descriptors import ESM
from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.graphs import ContactMap
from proteinggnnmetrics.kernels import LinearKernel
from proteinggnnmetrics.loaders import list_pdb_files, load_graphs
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates, Sequence
from proteinggnnmetrics.perturbations import Mutation
from proteinggnnmetrics.utils.functions import flatten_lists, remove_fragments


def get_longest_protein_dummy_sequence(pdb_files, cfg: DictConfig) -> int:
    seq = Sequence(n_jobs=cfg.compute.n_jobs)
    seq_normal = seq.fit_transform(
        pdb_files[
            cfg.experiments.proteins.not_perturbed.lower_bound : cfg.experiments.proteins.not_perturbed.upper_bound
            + 1
        ]
    )

    seq_perturbed = seq.fit_transform(
        pdb_files[
            cfg.experiments.proteins.perturbed.lower_bound : cfg.experiments.proteins.perturbed.upper_bound
            + 1
        ]
    )
    longest_sequence = max(
        max([len(protein.sequence) for protein in seq_normal]),
        max([len(protein.sequence) for protein in seq_perturbed]),
    )
    return longest_sequence


def execute_run(cfg, run):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    sampled_files = random.Random(run).sample(
        pdb_files, cfg.experiments.sample_size * 2
    )
    sampled_files = remove_fragments(sampled_files)
    midpoint = int(cfg.experiments.sample_size / 2)

    # Get longest proteins of proteins and proteins_perturbed
    dummy_longest = get_longest_protein_dummy_sequence(pdb_files, cfg)

    base_feature_steps = [
        (
            "coordinates",
            Coordinates(
                n_jobs=cfg.compute.n_jobs,
                granularity="CA",
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "contact_map",
            ContactMap(n_jobs=cfg.compute.n_jobs, verbose=cfg.debug.verbose),
        ),
        (
            "esm",
            ESM(
                size="M",
                n_jobs=cfg.compute.n_jobs,
                verbose=cfg.debug.verbose,
                longest_sequence=dummy_longest,
            ),
        ),
    ]
    base_feature_pipeline = pipeline.Pipeline(
        base_feature_steps, verbose=cfg.debug.verbose
    )
    proteins = base_feature_pipeline.fit_transform(sampled_files[midpoint:])
    results = list()
    for mutation in tqdm(
        np.arange(
            cfg.experiments.perturbations.mutation.min,
            cfg.experiments.perturbations.mutation.max,
            cfg.experiments.perturbations.mutation.step,
        ),
        position=1,
        leave=False,
        desc="Mutation probability",
    ):
        perturb_feature_steps = flatten_lists(
            [
                base_feature_steps[:1]
                + [
                    (
                        "mutate",
                        Mutation(
                            p_mutate=mutation,
                            random_state=np.random.RandomState(run),
                            n_jobs=cfg.compute.n_jobs,
                            verbose=cfg.debug.verbose,
                        ),
                    )
                ]
                + base_feature_steps[1:]
            ]
        )
        perturb_feature_pipeline = pipeline.Pipeline(
            perturb_feature_steps, verbose=cfg.debug.verbose
        )
        proteins_perturbed = perturb_feature_pipeline.fit_transform(
            sampled_files[:midpoint]
        )
        embeddings = np.array(
            [protein.embeddings["esm"] for protein in proteins]
        )
        embeddings_perturbed = np.array(
            [protein.embeddings["esm"] for protein in proteins_perturbed]
        )

        mmd_esm = MaximumMeanDiscrepancy(
            biased=True,
            squared=True,
            kernel=LinearKernel(n_jobs=cfg.compute.n_jobs),  # type: ignore
        ).compute(embeddings, embeddings_perturbed)

        graphs = load_graphs(proteins, graph_type="eps_graph")
        graphs_perturbed = load_graphs(
            proteins_perturbed, graph_type="eps_graph"
        )

        mmd_wl = MaximumMeanDiscrepancy(
            biased=True,
            squared=True,
            kernel=WeisfeilerLehmanKernel(
                n_jobs=cfg.compute.n_jobs,
                n_iter=5,
                verbose=cfg.debug.verbose,
                biased=True,
            ),  # type: ignore
        ).compute(graphs, graphs_perturbed)

        results.append(
            {"mmd_esm": mmd_esm, "mmd_wl": mmd_wl, "p_mutate": mutation}
        )

    results = pd.DataFrame(results).to_csv(
        here()
        / cfg.experiments.results
        / f"mmd_single_run_mutation_run_{run}.csv"
    )


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf_3")
def main(cfg: DictConfig):
    os.makedirs(here() / cfg.experiments.results, exist_ok=True)
    execute_run(cfg, run=cfg.experiments.n_runs)


if __name__ == "__main__":
    main()
