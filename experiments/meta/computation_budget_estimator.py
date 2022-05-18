#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""computation_budget_estimator.py

This script helps generate an estimate of the execution time of running the whole pipeline.

"""

import os
import random

import hydra
import pandas as pd
from omegaconf import DictConfig
from pyprojroot import here


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf")
def main(cfg):
    experiments = pd.read_csv(
        here() / "data" / "systematic" / "experimental_configurations.csv"
    )


if __name__ == "__main__":
    main()
