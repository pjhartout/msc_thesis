#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""other_kernels.py

Compute other kernels from mutations

"""

import os
import random

import hydra
import numpy as np
import pandas as pd
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pyprojroot import here
from torch import embedding
from tqdm import tqdm

from proteinggnnmetrics.descriptors import ESM
from proteinggnnmetrics.distance import MaximumMeanDiscrepancy
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.kernels import LinearKernel
from proteinggnnmetrics.loaders import list_pdb_files, load_graphs
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates, Sequence
from proteinggnnmetrics.perturbations import Mutation
from proteinggnnmetrics.utils.functions import (
    flatten_lists,
    remove_fragments,
    tqdm_joblib,
)


def execute_run(cfg, run):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    sampled_files = random.Random(run).sample(
        pdb_files, cfg.experiments.sample_size * 2
    )
    sampled_files = remove_fragments(sampled_files)
    midpoint = int(cfg.experiments.sample_size / 2)

    # Get longest proteins of proteins and proteins_perturbed
    base_feature_steps = [
        (
            "coordinates",
            Coordinates(
                granularity="CA",
                n_jobs=cfg.experiments.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "contact_map",
            ContactMap(
                n_jobs=cfg.experiments.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "epsilon_graph",
            EpsilonGraph(
                n_jobs=cfg.experiments.compute.n_jobs,
                epsilon=8,
                verbose=cfg.debug.verbose,
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
                            n_jobs=cfg.experiments.compute.n_jobs,
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

        graphs = load_graphs(proteins, graph_type="eps_graph")
        graphs_perturbed = load_graphs(
            proteins_perturbed, graph_type="eps_graph"
        )

        mmd_wl = MaximumMeanDiscrepancy(
            biased=True,
            squared=True,
            kernel=WeisfeilerLehmanKernel(
                n_jobs=cfg.experiments.compute.n_jobs,
                n_iter=5,
                verbose=cfg.debug.verbose,
                biased=True,
            ),  # type: ignore
        ).compute(graphs, graphs_perturbed)

        results.append({"mmd_wl": mmd_wl, "p_mutate": mutation})

    results = pd.DataFrame(results).to_csv(
        here()
        / cfg.experiments.results
        / f"mmd_single_run_mutation_run_{run}_other_kernels.csv"
    )
    return "ok"


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf_3")
def main(cfg: DictConfig):
    os.makedirs(here() / cfg.experiments.results, exist_ok=True)
    # for run in range(cfg.experiments.n_runs):
    #     execute_run(cfg, run)
    with tqdm_joblib(
        tqdm(
            desc="Execute runs in parallel",
            total=len(list(range(cfg.experiments.compute.n_parallel_runs))),
        )
    ) as progressbar:
        res = Parallel(n_jobs=cfg.experiments.compute.n_parallel_runs)(
            delayed(execute_run)(cfg, run)
            for run in range(cfg.experiments.n_runs)
        )
    print(res)


if __name__ == "__main__":
    main()