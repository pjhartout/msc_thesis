#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_script.py

This script is used to play around with the configuration files and compose them

"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf")
def conf(cfg: DictConfig) -> None:
    print("===== CONF =====")
    print(OmegaConf.to_yaml(cfg))


@hydra.main(config_path=str(here()) + "/conf/", config_name="config")
def config(cfg: DictConfig) -> None:
    print("===== CONFIG =====")
    print(OmegaConf.to_yaml(cfg))


def main() -> None:
    conf()
    config()


if __name__ == "__main__":
    main()
