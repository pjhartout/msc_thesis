#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""exp_2.py

The goal of this experiment is to investigate the effect of twisting on MMD features derived from TDA.

"""

import logging
import os
import pickle
import random
from typing import Dict, List

import hydra
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from fastwlk.kernel import WeisfeilerLehmanKernel
from gtda import pipeline
from gtda.homology import VietorisRipsPersistence
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinggnnmetrics.descriptors import (
    LaplacianSpectrum,
    RamachandranAngles,
    TopologicalDescriptor,
)
from proteinggnnmetrics.distance import (
    MaximumMeanDiscrepancy,
    PearsonCorrelation,
    SpearmanCorrelation,
)
from proteinggnnmetrics.graphs import ContactMap, EpsilonGraph
from proteinggnnmetrics.kernels import LinearKernel, PersistenceFisherKernel
from proteinggnnmetrics.loaders import (
    list_pdb_files,
    load_descriptor,
    load_graphs,
)
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.perturbations import GaussianNoise, Twist
from proteinggnnmetrics.protein import Protein
from proteinggnnmetrics.utils.debug import SamplePoints, measure_memory, timeit
from proteinggnnmetrics.utils.functions import (
    distribute_function,
    flatten_lists,
    remove_fragments,
    tqdm_joblib,
)

log = logging.getLogger(__name__)


def execute_twist(
    twist: float,
    base_feature_steps,
    cfg,
    sampled_files,
    midpoint,
    proteins: List[Protein],
) -> Dict:
    log.info(f"Starting experiment with twist {twist}")
    perturb_feature_steps = flatten_lists(
        [
            base_feature_steps[:1]
            + [
                (
                    "twist",
                    Twist(
                        alpha=twist,
                        random_state=42,
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

    ramachandran = [protein.phi_psi_angles for protein in proteins]
    ramachandran_perturbed = [
        protein.phi_psi_angles for protein in proteins_perturbed
    ]

    mmd_rama = MaximumMeanDiscrepancy(
        biased=True,
        squared=True,
        kernel=LinearKernel(
            n_jobs=cfg.experiments.compute.n_jobs
        ),  # type: ignore
    ).compute(ramachandran, ramachandran_perturbed)
    return {"mmd_rama": mmd_rama, "twist": twist}


def execute_run(cfg, run):
    log.info(f"Start run {run}")
    os.makedirs(here() / cfg.experiments.results, exist_ok=True)
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    pdb_files = remove_fragments(pdb_files)
    log.info(f"Number of files left {len(pdb_files)}")
    sampled_files = random.Random(run).sample(
        pdb_files, cfg.experiments.sample_size * 2
    )
    midpoint = int(len(sampled_files) / 2)

    base_feature_steps = [
        (
            "coodinates",
            Coordinates(
                granularity="CA",
                n_jobs=cfg.experiments.compute.n_jobs,
                verbose=cfg.debug.verbose,
            ),
        ),
        (
            "ramachandran",
            RamachandranAngles(
                from_pdb=True,
                n_jobs=cfg.experiments.compute.n_jobs,
                verbose=cfg.debug.verbose,
                n_bins=100,
                bin_range=(-np.pi, np.pi),
            ),
        ),
    ]

    base_feature_pipeline = pipeline.Pipeline(
        base_feature_steps, verbose=cfg.debug.verbose
    )
    proteins = base_feature_pipeline.fit_transform(sampled_files[midpoint:],)
    log.info(f"Starting twist execution for run {run}")
    results = distribute_function(
        execute_twist,
        np.arange(
            cfg.experiments.perturbations.twist.min,
            cfg.experiments.perturbations.twist.max,
            cfg.experiments.perturbations.twist.step,
        ),
        n_jobs=cfg.experiments.compute.n_pipelines,
        tqdm_label="Distribute twists",
        show_tqdm=cfg.debug.verbose,
        total=len(
            np.arange(
                cfg.experiments.perturbations.twist.min,
                cfg.experiments.perturbations.twist.max,
                cfg.experiments.perturbations.twist.step,
            )
        ),
        base_feature_steps=base_feature_steps,
        cfg=cfg,
        sampled_files=sampled_files,
        midpoint=midpoint,
        proteins=proteins,
    )

    log.info("Dumping results")
    results = pd.DataFrame(results).to_csv(
        here()
        / cfg.experiments.results
        / f"mmd_single_run_twist_{run}_ramachandran.csv"
    )


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf_2")
def main(cfg: DictConfig):
    log.info("Info level message")
    os.makedirs(here() / cfg.experiments.results, exist_ok=True)
    with tqdm_joblib(
        tqdm(
            desc="Execute runs in parallel",
            total=len(list(range(cfg.experiments.compute.n_parallel_runs))),
        )
    ) as progressbar:
        Parallel(n_jobs=cfg.experiments.compute.n_parallel_runs)(
            delayed(execute_run)(cfg, run)
            for run in range(cfg.experiments.n_runs)
        )


if __name__ == "__main__":
    main()
