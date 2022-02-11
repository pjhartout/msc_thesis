#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paths.py

Provides global variables to other scripts used in the repository.
"""
from pathlib import Path

from pyprojroot import here

REPO_HOME = here()
DATA_HOME = "/local0/scratch/phartout"

HUMAN_PROTEOME = DATA_HOME / Path("data/UP000005640_9606_HUMAN_v2/")

HUMAN_PROTEOME_CA_GRAPHS = REPO_HOME / Path(
    "data/UP000005640_9606_HUMAN_v2_CA_GRAPHS/"
)
HUMAN_PROTEOME_CA_CONTACT_MAP = REPO_HOME / Path(
    "data/UP000005640_9606_HUMAN_v2_CA_CONTACT_MAP/"
)
