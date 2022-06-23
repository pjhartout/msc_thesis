#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tda_pc_batch_job_generator.py

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
        here() / "experiments/systematic/tda_pc" / "tda_pc_job_array.sh",
        "w",
    )  # write mode
    wl_pc.write(f"#!/bin/bash \n")
    wl_pc.write(f"\n\n")
    wl_pc.write(f"cd /home/phartout/Documents/Git/msc_thesis/ \n")
    wl_pc.write(f"export PATH=/home/phartout/.anaconda3/bin:$PATH\n\n")
    slurm_string = "srun --cpus-per-task 20 --mem-per-cpu 50G poetry run python experiments/systematic/tda_pc/tda_pc.py"

    perturbations = [
        "twist",
        "shear",
        "taper",
        "gaussian_noise",
    ]

    for perturbation in perturbations:
        job_param = f"{slurm_string} +perturbation={perturbation} \n"
        wl_pc.write(job_param)
        wl_pc.write(build_fail_string(job_param))

    wl_pc.write(f'echo "Done"\n')
    wl_pc.close()


if __name__ == "__main__":
    main()
