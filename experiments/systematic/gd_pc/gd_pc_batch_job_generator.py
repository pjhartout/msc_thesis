#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""gd_pc_batch_job_generator.py

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
        here() / "experiments/systematic/gd_pc" / "gd_pc_job_array.sh",
        "w",
    )  # write mode
    wl_pc.write(f"#!/bin/bash \n")
    wl_pc.write(f"\n\n")
    wl_pc.write(f"cd /home/phartout/Documents/Git/msc_thesis/ \n")
    wl_pc.write(f"export PATH=/home/phartout/.anaconda3/bin:$PATH\n\n")
    slurm_string = "srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py"

    descriptors = [
        "degree_histogram",
        "clustering_histogram",
        "laplacian_spectrum_histogram",
    ]
    pc_descriptors = [
        "distance_histogram",
        "dihedral_angles_histogram",
    ]
    perturbations = [
        "twist",
        "shear",
        "taper",
        "gaussian_noise",
        "mutation",
    ]

    for descriptor in pc_descriptors:
        for perturbation in perturbations:
            job_param = f"{slurm_string} +descriptor={descriptor} +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation={perturbation} \n"
            wl_pc.write(job_param)
            wl_pc.write(build_fail_string(job_param))

    for eps in cfg.meta.representations[0]["eps_graph"]:
        for descriptor in descriptors:
            for perturbation in perturbations:
                job_param = f"{slurm_string} +descriptor={descriptor} +graph_type=eps_graph +graph_extraction_parameter={eps} +perturbation={perturbation} \n"
                wl_pc.write(job_param)
                wl_pc.write(build_fail_string(job_param))

    for k in cfg.meta.representations[1]["knn_graph"]:
        for descriptor in descriptors:
            for perturbation in perturbations:
                job_param = f"{slurm_string} +descriptor={descriptor} +graph_type=knn_graph +graph_extraction_parameter={k} +perturbation={perturbation}\n"
                wl_pc.write(job_param)
                wl_pc.write(build_fail_string(job_param))

    wl_pc.write(f'echo "Done"\n')
    wl_pc.close()
    print(
        f"Wrote {here() / 'experiments/systematic/gc_pc' / 'gc_pc_job_array.sh'} "
    )


if __name__ == "__main__":
    main()
