#!/bin/bash 


cd /home/phartout/Documents/Git/msc_thesis/ 
export PATH=/home/phartout/.anaconda3/bin:$PATH

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=distance_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=twist 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=distance_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=twist 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=distance_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=shear 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=distance_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=shear 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=distance_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=taper 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=distance_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=taper 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=distance_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=gaussian_noise 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=distance_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=gaussian_noise 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=distance_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=mutation 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=distance_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=mutation 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=dihedral_angles_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=twist 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=dihedral_angles_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=twist 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=dihedral_angles_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=shear 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=dihedral_angles_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=shear 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=dihedral_angles_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=taper 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=dihedral_angles_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=taper 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=dihedral_angles_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=gaussian_noise 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=dihedral_angles_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=gaussian_noise 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=dihedral_angles_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=mutation 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=dihedral_angles_histogram +graph_type=pc_descriptor +graph_extraction_parameter=1 +perturbation=mutation 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=twist 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=twist 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=shear 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=shear 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=taper 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=taper 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=mutation 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=mutation 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=twist 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=twist 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=shear 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=shear 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=taper 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=taper 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=mutation 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=mutation 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=twist 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=twist 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=shear 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=shear 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=taper 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=taper 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=mutation 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=mutation 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=twist 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=twist 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=shear 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=shear 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=taper 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=taper 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=gaussian_noise 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=gaussian_noise 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=mutation 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=mutation 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=twist 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=twist 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=shear 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=shear 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=taper 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=taper 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=gaussian_noise 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=gaussian_noise 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=mutation 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=mutation 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=twist 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=twist 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=shear 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=shear 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=taper 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=taper 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=gaussian_noise 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=gaussian_noise 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=mutation 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=16 +perturbation=mutation 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=twist 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=twist 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=shear 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=shear 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=taper 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=taper 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=gaussian_noise 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=gaussian_noise 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=mutation 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=mutation 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=twist 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=twist 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=shear 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=shear 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=taper 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=taper 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=gaussian_noise 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=gaussian_noise 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=mutation 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=mutation 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=twist 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=twist 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=shear 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=shear 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=taper 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=taper 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=gaussian_noise 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=gaussian_noise 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=mutation 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=32 +perturbation=mutation 
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=twist

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=twist
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=shear

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=shear
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=taper

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=taper
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=gaussian_noise

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=gaussian_noise
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=mutation

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=mutation
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=twist

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=twist
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=shear

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=shear
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=taper

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=taper
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=gaussian_noise

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=gaussian_noise
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=mutation

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=mutation
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=twist

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=twist
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=shear

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=shear
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=taper

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=taper
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=gaussian_noise

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=gaussian_noise
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=mutation

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=2 +perturbation=mutation
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=twist

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=twist
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=shear

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=shear
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=taper

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=taper
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=gaussian_noise

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=gaussian_noise
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=mutation

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=mutation
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=twist

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=twist
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=shear

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=shear
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=taper

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=taper
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=gaussian_noise

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=gaussian_noise
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=mutation

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=mutation
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=twist

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=twist
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=shear

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=shear
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=taper

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=taper
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=gaussian_noise

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=gaussian_noise
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=mutation

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=6 +perturbation=mutation
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=twist

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=twist
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=shear

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=shear
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=taper

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=taper
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=mutation

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=degree_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=mutation
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=twist

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=twist
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=shear

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=shear
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=taper

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=taper
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=mutation

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=clustering_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=mutation
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=twist

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=twist
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=shear

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=shear
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=taper

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=taper
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=gaussian_noise
" >> failed_jobs.txt
fi

srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=mutation

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 50 --mem-per-cpu 7G poetry run python experiments/systematic/gd_pc/gd_pc.py +descriptor=laplacian_spectrum_histogram +graph_type=knn_graph +graph_extraction_parameter=8 +perturbation=mutation
" >> failed_jobs.txt
fi

echo "Done"
