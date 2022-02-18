#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""paths.py

Provides global variables to other scripts used in the repository.
"""

import os
from pathlib import Path, PosixPath

from pyprojroot import here

REPO_HOME = here()

FAST_DATA_HOME = "/local0/scratch/phartout"


def set_data_home() -> PosixPath:
    """Context-aware setting of data path to leverage cluster SSDs if able.

    Returns:
        PosixPath: path where the data should be located.
    """
    if os.path.isdir(FAST_DATA_HOME):
        # Use fast SSD in cluster
        print(f"Data path: {FAST_DATA_HOME}")
        return FAST_DATA_HOME
    else:
        # Switch to default. Make sure repo is on fast SSD.
        data_home = REPO_HOME / "data"
        print(f"Data path: {data_home}")
        return data_home


DATA_HOME = set_data_home()
HUMAN_PROTEOME = DATA_HOME / "UP000005640_9606_HUMAN_v2"

HUMAN_PROTEOME_CA_GRAPHS = DATA_HOME / Path(
    "UP000005640_9606_HUMAN_v2_CA_GRAPHS"
)
HUMAN_PROTEOME_CA_CONTACT_MAP = DATA_HOME / Path(
    "data/UP000005640_9606_HUMAN_v2_CA_CONTACT_MAP/"
)
