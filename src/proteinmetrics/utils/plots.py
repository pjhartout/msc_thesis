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
    mpl.rcParams["font.serif"] = ["Palatino"]
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["axes.unicode_minus"] = False


def setup_annotations(
    p, title: str, x_label: str, y_label: str, legend_title: str,
):
    p.set_xlabel(x_label)
    p.set_ylabel(y_label)
    plt.legend(title=legend_title, loc="best")
    plt.title(title)
    plt.tight_layout()
    return p
