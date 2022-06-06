#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""gd_graphs_batch_job_generator.py

This script generates the series of jobs that need to be run on the cluster

"""


import hydra
from pyprojroot import here


def build_fail_string(job_param):
    return f"""
if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED {job_param}" >> failed_jobs.txt
fi\n
"""


@hydra.main(
    version_base=None, config_path=str(here()) + "/conf/", config_name="conf"
)
def main(cfg):
    wl_pc = open(
        here() / "experiments/systematic/gd_graphs" / "gd_graphs_job_array.sh",
        "w",
    )  # write mode
    wl_pc.write(f"#!/bin/bash \n")
    wl_pc.write(f"\n\n")
    wl_pc.write(f"cd /home/phartout/Documents/Git/msc_thesis/ \n")
    wl_pc.write(f"export PATH=/home/phartout/.anaconda3/bin:$PATH\n\n")
    slurm_string = f"srun --cpus-per-task {cfg.compute.n_jobs*cfg.compute.n_parallel_perturb} --mem-per-cpu 10G poetry run python experiments/systematic/gd_graphs/gd_graphs.py"

    descriptors = [
        "degree_histogram",
        "clustering_histogram",
        "laplacian_spectrum_histogram",
    ]
    perturbations = [
        "remove_edges",
        "add_edges",
        "rewire_edges",
    ]

    for descriptor in descriptors:
        for eps in cfg.meta.representations[0]["eps_graph"]:
            for perturbation in perturbations:
                job_param = f"{slurm_string} +descriptor={descriptor} +graph_type=eps_graph +graph_extraction_parameter={eps} +perturbation={perturbation} \n"
                wl_pc.write(job_param)
                wl_pc.write(build_fail_string(job_param))

        for k in cfg.meta.representations[1]["knn_graph"]:
            for perturbation in perturbations:
                job_param = f"{slurm_string} +descriptor={descriptor} +graph_type=knn_graph +graph_extraction_parameter={k} +perturbation={perturbation} \n"
                wl_pc.write(job_param)
                wl_pc.write(build_fail_string(job_param))

    wl_pc.write(f'echo "Done"\n')
    wl_pc.close()
    print(
        f"Wrote {here() / 'experiments/systematic/gd_graphs' / 'gd_graphs_job_array.sh'} "
    )


if __name__ == "__main__":
    main()
