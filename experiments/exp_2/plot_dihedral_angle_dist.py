#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot_dihedral_angle_dist.py

Plots an example of dihedral angle distribution.

"""

import os
from pathlib import Path

import hydra
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser, PPBuilder
from gtda import pipeline
from omegaconf import DictConfig
from pyprojroot import here

from proteinggnnmetrics.descriptors import RamachandranAngles
from proteinggnnmetrics.loaders import list_pdb_files
from proteinggnnmetrics.paths import HUMAN_PROTEOME
from proteinggnnmetrics.pdb import Coordinates
from proteinggnnmetrics.utils.functions import (
    distribute_function,
    flatten_lists,
    remove_fragments,
    tqdm_joblib,
)
from proteinggnnmetrics.utils.plots import (
    setup_annotations,
    setup_plotting_parameters,
)

setup_plotting_parameters()


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf_2")
def main(cfg: DictConfig):
    os.makedirs(here() / cfg.experiments.results / "images", exist_ok=True)

    pdb_file = list_pdb_files(HUMAN_PROTEOME)[2]

    protein = Coordinates(granularity="CA", n_jobs=1).get_atom_coordinates(
        pdb_file
    )

    parser = PDBParser()
    structure = parser.get_structure(protein.path.stem, protein.path)  # type: ignore

    angles = dict()
    for idx_model, model in enumerate(structure):
        polypeptides = PPBuilder().build_peptides(model)
        for idx_poly, poly in enumerate(polypeptides):
            angles[f"{idx_model}_{idx_poly}"] = poly.get_phi_psi_list()

    phi = np.array(flatten_lists(angles.values()), dtype=object)[:, 0].astype(
        float
    )
    psi = np.array(flatten_lists(angles.values()), dtype=object)[:, 1].astype(
        float
    )
    phi = phi[~np.isnan(phi)]
    psi = psi[~np.isnan(psi)]

    palette = sns.color_palette("tab10")[1]
    df_plot = pd.Series(phi, name="Phi")
    p = sns.displot(
        df_plot, kind="kde", rug=True, color=palette, height=3, aspect=1
    )

    plt.title("Phi Distribution")
    plt.xlabel("Phi (rad)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(
        here() / cfg.experiments.results / "images/dehedral_dist_phi.png"
    )

    df_plot = pd.Series(phi, name="Psi")
    p = sns.displot(
        df_plot, kind="kde", rug=True, color=palette, height=3, aspect=1
    )
    plt.title("Psi Distribution")
    plt.xlabel("Psi (rad)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(
        here() / cfg.experiments.results / "images/dehedral_dist_psi.png"
    )


if __name__ == "__main__":
    main()
