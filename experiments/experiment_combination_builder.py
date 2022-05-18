#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""experiment_combination_builder.py

This script generates all the combination of experiments to compute

"""


from typing import Dict, List

import hydra
import numpy as np
import omegaconf
import pandas as pd
from omegaconf import DictConfig
from pyprojroot import here

from proteinggnnmetrics.utils.functions import make_dir

# Constants
point_cloud_perturbations = ["shear", "taper", "twist", "gaussian"]


def cartesian_product_dct(d):
    index = pd.MultiIndex.from_product(d.values(), names=d.keys())
    return pd.DataFrame(index=index).reset_index()


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def cartesian_product_multi(*dfs):
    idx = cartesian_product(*[np.ogrid[: len(df)] for df in dfs])
    return pd.DataFrame(
        np.column_stack([df.values[idx[:, i]] for i, df in enumerate(dfs)])
    )


def from_list(lst: omegaconf.ListConfig, colname: str) -> pd.DataFrame:
    return pd.DataFrame([lst]).T.rename(columns={0: colname})


def from_nested_list(
    lst: omegaconf.ListConfig, colnames: List[str]
) -> pd.DataFrame:
    opts = list()
    for elem in lst:
        for v in [*elem.values()][0]:
            opts.append(str([*elem][0]) + "_" + str(v))
    return pd.DataFrame(opts, columns=colnames)


def kernel_combinations(lst: omegaconf.ListConfig) -> Dict[str, pd.DataFrame]:
    # Build a dictionary where each kernel gets a df
    kernel_dict = dict()
    for kernel in lst:
        df_params = pd.DataFrame()
        for params in kernel.values():
            for param_dict in params:
                df_params = df_params.join(
                    pd.DataFrame(param_dict.values()).T.rename(
                        columns={0: next(iter(param_dict.keys()))}
                    ),
                    how="outer",
                )
        kernel_dict[next(iter(kernel.keys()))] = df_params

    # get all combinations for each kernel in case multiple parameters are mentioned
    for opt in kernel_dict.items():
        opt_dict = dict()
        if opt[1].columns.nunique() > 1:
            for col in opt[1].columns:
                opt_dict[col] = opt[1][col].tolist()
            kernel_dict[opt[0]] = cartesian_product_dct(opt_dict)

    return kernel_dict


def tda_filter(opts):
    # Filter relevant options for TDA experiments
    tda_experiments = opts[
        (opts["descriptors"] == "persistence_diagram")
        & (opts["representations"] == "point_cloud_CA")
        & (opts["kernel"] == "persistence_fisher")
        & (opts["perturbations"].isin(point_cloud_perturbations))
    ]

    # To keep the logic of the filters clear, we just remove the tda options
    # and add back in the ones we have filtered out above
    # TODO: Consider improving logic to avoid this step.

    opts = opts[
        (opts["descriptors"] != "persistence_diagram")
        & (opts["representations"] != "point_cloud_CA")
        & (opts["kernel"] != "persistence_fisher")
    ]
    return pd.concat([opts, tda_experiments])


def weisfeiler_lehman_filter(opts):
    # Weisfeiler-Lehman kernel van only be computed on graphs
    graph_experiments = opts[
        (opts["kernel"] == "weisfeiler-lehman")
        & opts["representations"].str.contains("graphs")
        & (
            opts["descriptors"] == "degree_histogram"
        )  # we filter out other reps because they are not supported
    ]

    opts = opts[(opts["kernel"] != "weisfeiler-lehman")]
    return pd.concat([opts, graph_experiments])


def esm_filter(opts):
    # ESM can only be computed on sequences
    esm_experiments = opts[
        (opts["descriptors"] == "esm")
        & (
            opts["representations"] == "sequence_"
        )  # we filter out other reps because they are not supported
        & (opts["perturbations"] == "mutate")
    ]

    opts = opts[
        (opts["descriptors"] != "esm")
        & (opts["representations"] != "sequence_")
    ]
    return pd.concat([opts, esm_experiments])


def angles_filter(opts):
    # angles can only be computed on sequences
    angles_experiments = opts[
        (opts["descriptors"] == "angles")
        & (
            opts["representations"] == "point_cloud_backbone"
        )  # we filter out other reps because they are not supported
        & (opts.perturbations.isin(point_cloud_perturbations))
    ]

    opts = opts[
        (opts["descriptors"] != "angles")
        & (opts["representations"] != "point_cloud_backbone")
    ]
    return pd.concat([opts, angles_experiments])


@hydra.main(config_path=str(here()) + "/conf/", config_name="conf")
def main(cfg):
    options_to_consider = [k for k in cfg.meta.keys() if "_" not in k]

    options_to_consider_dict = {k: cfg.meta[k] for k in options_to_consider}
    organisms = from_list(cfg.meta.organisms, "organisms")
    perturbations = from_list(cfg.meta.perturbations, "perturbations")
    descriptors = from_list(cfg.meta.descriptors, "descriptors")
    representations = from_nested_list(
        cfg.meta.representations, ["representations"]
    )
    kernels = kernel_combinations(cfg.meta.kernels)

    # Get all combinations of all options except kernels
    all_options_wo_kernel = cartesian_product_multi(
        *[organisms, perturbations, descriptors, representations]
    )
    all_options_wo_kernel.columns = [
        "organisms",
        "perturbations",
        "descriptors",
        "representations",
    ]

    all_opts = pd.DataFrame()
    for k in kernels:
        all_opts = pd.concat(
            [
                all_opts,
                all_options_wo_kernel.merge(kernels[k], how="cross").assign(
                    kernel=k
                ),
            ]
        )

    all_opts = all_opts.reset_index(drop=True)

    # Now the filtering starts.
    all_opts = tda_filter(all_opts)
    all_opts = weisfeiler_lehman_filter(all_opts)
    all_opts = esm_filter(all_opts)
    all_opts = angles_filter(all_opts)
    all_opts = all_opts.reset_index(drop=True)

    # Remove params column since it's a dummy placeholder for a parameter-free
    # kernel
    all_opts = all_opts.drop(columns="params")

    # Create directory
    make_dir(str(here() / "systematic"))
    all_opts.to_csv(
        here() / "data" / "systematic" / "experimental_configurations.csv"
    )


if __name__ == "__main__":
    main()
