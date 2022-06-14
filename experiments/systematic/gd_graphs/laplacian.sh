#!/bin/bash 


cd /home/phartout/Documents/Git/msc_thesis/ 
export PATH=/home/phartout/.anaconda3/bin:$PATH

srun --cpus-per-task 30 --mem-per-cpu 10G poetry run python experiments/systematic/gd_graphs/gd_graphs.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=remove_edges 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 30 --mem-per-cpu 10G poetry run python experiments/systematic/gd_graphs/gd_graphs.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=remove_edges 
" >> failed_jobs.txt
fi

srun --cpus-per-task 30 --mem-per-cpu 10G poetry run python experiments/systematic/gd_graphs/gd_graphs.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=add_edges 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 30 --mem-per-cpu 10G poetry run python experiments/systematic/gd_graphs/gd_graphs.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=add_edges 
" >> failed_jobs.txt
fi

srun --cpus-per-task 30 --mem-per-cpu 10G poetry run python experiments/systematic/gd_graphs/gd_graphs.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=rewire_edges 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 30 --mem-per-cpu 10G poetry run python experiments/systematic/gd_graphs/gd_graphs.py +descriptor=laplacian_spectrum_histogram +graph_type=eps_graph +graph_extraction_parameter=8 +perturbation=rewire_edges 
" >> failed_jobs.txt
fi


echo "Done"
