#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""make_gif.py

Make GIF from collection of images

"""

import os
import re
from typing import Dict

import hydra
import imageio
from gtda import pipeline
from natsort import natsorted
from omegaconf import DictConfig
from pyprojroot import here
from tqdm import tqdm

from proteinmetrics.graphs import ContactMap, EpsilonGraph
from proteinmetrics.loaders import list_pdb_files
from proteinmetrics.paths import HUMAN_PROTEOME
from proteinmetrics.pdb import Coordinates
from proteinmetrics.perturbations import GaussianNoise, Shear, Taper, Twist
from proteinmetrics.utils.debug import measure_memory, timeit


def build_gaussian(cfg: DictConfig):
    images_fnames = list()
    images = list()
    for fname in tqdm(os.listdir(here() / cfg.imaging.gaussian_noise_path)):
        images_fnames.append(fname)
    images_fnames.sort(key=lambda f: int(re.sub("\D", "", f)))
    for fname in tqdm(images_fnames):
        images.append(
            imageio.imread(here() / cfg.imaging.gaussian_noise_path / fname)
        )
    imageio.mimsave(here() / cfg.imaging.gifs / "gaussian_noise.gif", images)


def build_twist(cfg: DictConfig):
    images_fnames = list()
    images = list()
    for fname in tqdm(os.listdir(here() / cfg.imaging.twist_path)):
        images_fnames.append(fname)
    images_fnames = natsorted(images_fnames)
    for fname in tqdm(images_fnames):
        images.append(imageio.imread(here() / cfg.imaging.twist_path / fname))
    imageio.mimsave(here() / cfg.imaging.gifs / "twist.gif", images)


def build_shear(cfg: DictConfig):
    images_fnames = list()
    images = list()
    for fname in tqdm(os.listdir(here() / cfg.imaging.shear_path)):
        images_fnames.append(fname)
    images_fnames = natsorted(images_fnames)
    for fname in tqdm(images_fnames):
        images.append(imageio.imread(here() / cfg.imaging.shear_path / fname))
    imageio.mimsave(here() / cfg.imaging.gifs / "shear.gif", images)


def build_taper(cfg: DictConfig):
    images_fnames = list()
    images = list()
    for fname in tqdm(os.listdir(here() / cfg.imaging.taper_path)):
        images_fnames.append(fname)
    images_fnames = natsorted(images_fnames)
    for fname in tqdm(images_fnames):
        images.append(imageio.imread(here() / cfg.imaging.taper_path / fname))
    imageio.mimsave(here() / cfg.imaging.gifs / "taper.gif", images)


def build_mutation(cfg: DictConfig):
    images_fnames = list()
    images = list()
    for fname in tqdm(os.listdir(here() / cfg.imaging.mutation_path)):
        images_fnames.append(fname)
    images_fnames = natsorted(images_fnames)
    for fname in tqdm(images_fnames):
        images.append(
            imageio.imread(here() / cfg.imaging.mutation_path / fname)
        )
    imageio.mimsave(here() / cfg.imaging.gifs / "mutation.gif", images)


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf")
def main(cfg: DictConfig):
    build_gaussian(cfg)
    build_twist(cfg)
    build_shear(cfg)
    build_taper(cfg)
    build_mutation(cfg)


if __name__ == "__main__":
    main()
