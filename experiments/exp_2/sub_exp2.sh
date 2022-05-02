env2lmod
module load gcc/8.2.0
module load eth_proxy
module load git/2.31.1
module load python/3.9.9
cd /cluster/scratch/phartout/msc_thesis/
poetry run python /cluster/scratch/phartout/msc_thesis/experiments/exp_2/exp_2.py
