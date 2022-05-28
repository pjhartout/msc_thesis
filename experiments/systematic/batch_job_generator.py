#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""batch_job_generator.py

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
        here() / "experiments/systematic" / "wl_pc_job_array.sh",
        "w",
    )  # write mode
    wl_pc.write(f"#!/bin/bash \n")
    wl_pc.write(f"\n\n")
    wl_pc.write(f"cd /home/phartout/Documents/Git/msc_thesis/ \n")
    wl_pc.write(f"export PATH=/home/phartout/.anaconda3/bin:$PATH\n\n")
    slurm_string = "srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py"

    for n_iters in cfg.meta.kernels[3]["weisfeiler-lehman"][0]["n_iter"]:
        for eps in cfg.meta.representations[0]["eps_graph"]:
            job_param = f"{slurm_string} +n_iter={n_iters} +graph_type=eps_graph +graph_extraction_parameter={eps} \n"
            wl_pc.write(job_param)
            wl_pc.write(build_fail_string(job_param))

        for k in cfg.meta.representations[1]["knn_graph"]:
            job_param = f"{slurm_string} +n_iter={n_iters} +graph_type=knn_graph +graph_extraction_parameter={k} \n"
            wl_pc.write(job_param)
            wl_pc.write(build_fail_string(job_param))

    wl_pc.write(f'echo "Done"\n')
    wl_pc.close()


if __name__ == "__main__":
    main()
