# -*- coding: utf-8 -*-

"""plots.py

This file contains plotting utilities

"""

from typing import Tuple
from unittest import result

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt


def setup_plotting_parameters(
    resolution: int = 600, size: Tuple[float, float] = (7.8, 5.8)
) -> None:
    plt.rcParams["figure.figsize"] = size
    plt.rcParams["savefig.dpi"] = resolution
    mpl.rcParams["font.family"] = "serif"
    cmfont = font_manager.FontProperties(
        fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf"
    )
    mpl.rcParams["font.serif"] = cmfont.get_name()
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["axes.unicode_minus"] = False
