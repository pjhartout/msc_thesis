#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""boilerplate.py

Boilerplate code to test out hydra.

"""

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here


@hydra.main(
    config_path=str(here()) + "/conf/experiment",
    config_name="epsilon_experiment.yaml",
)
def func(cfg: DictConfig):
    """Boilerplate function."""
    logging.info("Subconfig")
    logging.info(f"cfg: {OmegaConf.to_yaml(cfg)}")


@hydra.main(config_path=str(here()) + "/conf", config_name="config.yaml")
def main(cfg):
    working_dir = os.getcwd()
    logging.info(f"Working directory: {working_dir}")
    logging.info(f"Config: {OmegaConf.to_yaml(cfg)}")
    logging.info("Calling function with separate config...")
    func(cfg)


if __name__ == "__main__":
    main()
