#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""data_index_builder.py

This script generates the paths to the proteins that we are going to use in our experiments.

"""

import os
import random

import hydra
import pandas as pd
from omegaconf import DictConfig
from pyprojroot import here

from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.utils.functions import make_dir


@hydra.main(
    version_base=None, config_path=str(here()) + "/conf/", config_name="conf"
)
def main(cfg):
    pdb_files = list_pdb_files(HUMAN_PROTEOME)
    make_dir(here() / cfg.meta.splits_dir)
    for run in range(cfg.meta.n_runs):
        sampled_files = random.Random(run).sample(
            pdb_files, cfg.meta.sample_size * 2
        )
        half = cfg.meta.sample_size
        sampled_files = [file.name for file in sampled_files]
        df_unperturbed = pd.DataFrame(sampled_files[:half], columns=["pdb_id"])
        df_perturbed = pd.DataFrame(sampled_files[half:], columns=["pdb_id"])

        df_unperturbed.to_csv(
            os.path.join(
                here() / cfg.meta.splits_dir,
                f"data_split_{run}_unperturbed.csv",
            ),
            index=False,
        )
        df_perturbed.to_csv(
            os.path.join(
                here() / cfg.meta.splits_dir,
                f"data_split_{run}_perturbed.csv",
            ),
            index=False,
        )

    print("Data for different runs is set")


if __name__ == "__main__":
    main()
