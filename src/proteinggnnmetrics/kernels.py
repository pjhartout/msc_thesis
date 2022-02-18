# -*- coding: utf-8 -*-

"""filename.py

***file description***

"""

import os
from abc import ABCMeta
from typing import Any

import numpy as np


class Kernel(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self):
        pass

    def transform(self, X: Any) -> np.ndarray:
        """Apply transformation to apply kernel to X
        """
        pass
