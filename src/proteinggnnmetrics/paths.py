#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""paths.py

Provides global variables to other scripts used in the repository.
"""

import os
from pathlib import Path, PosixPath

from pyprojroot import here

REPO_HOME = here()

# Location of fast ssd cache disk
FAST_DATA_HOME = "/local0/scratch/phartout"
FORCE_NETWORK_STORAGE = True


def set_data_home() -> PosixPath:
    """Context-aware setting of data path to leverage cluster SSDs if able.

    Returns:
        PosixPath: path where the data should be located.
    """
    if os.path.isdir(FAST_DATA_HOME) and not FORCE_NETWORK_STORAGE:
        # Use fast SSD in cluster
        print(f"Data path: {FAST_DATA_HOME}")
        return PosixPath(FAST_DATA_HOME) / "data"
    else:
        # Switch to default. Make sure repo is on fast SSD.
        data_home = REPO_HOME / "data"
        print(f"Data path: {data_home}")
        return PosixPath(data_home)


DATA_HOME = set_data_home()
CACHE_DIR = DATA_HOME / ".cache"
HUMAN_PROTEOME = DATA_HOME / "UP000005640_9606_HUMAN_v2"
ECOLI_PROTEOME = DATA_HOME / "UP000000625_83333_ECOLI_v2"
