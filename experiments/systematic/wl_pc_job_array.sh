#!/bin/bash 


cd /home/phartout/Documents/Git/msc_thesis/ 
export PATH=/home/phartout/.anaconda3/bin:$PATH

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=eps_graph +graph_extraction_parameter=8 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=eps_graph +graph_extraction_parameter=8 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=eps_graph +graph_extraction_parameter=16 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=eps_graph +graph_extraction_parameter=16 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=eps_graph +graph_extraction_parameter=32 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=eps_graph +graph_extraction_parameter=32 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=knn_graph +graph_extraction_parameter=2 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=knn_graph +graph_extraction_parameter=2 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=knn_graph +graph_extraction_parameter=6 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=knn_graph +graph_extraction_parameter=6 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=knn_graph +graph_extraction_parameter=8 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=1 +graph_type=knn_graph +graph_extraction_parameter=8 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=eps_graph +graph_extraction_parameter=8 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=eps_graph +graph_extraction_parameter=8 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=eps_graph +graph_extraction_parameter=16 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=eps_graph +graph_extraction_parameter=16 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=eps_graph +graph_extraction_parameter=32 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=eps_graph +graph_extraction_parameter=32 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=knn_graph +graph_extraction_parameter=2 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=knn_graph +graph_extraction_parameter=2 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=knn_graph +graph_extraction_parameter=6 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=knn_graph +graph_extraction_parameter=6 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=knn_graph +graph_extraction_parameter=8 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=2 +graph_type=knn_graph +graph_extraction_parameter=8 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=eps_graph +graph_extraction_parameter=8 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=eps_graph +graph_extraction_parameter=8 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=eps_graph +graph_extraction_parameter=16 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=eps_graph +graph_extraction_parameter=16 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=eps_graph +graph_extraction_parameter=32 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=eps_graph +graph_extraction_parameter=32 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=knn_graph +graph_extraction_parameter=2 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=knn_graph +graph_extraction_parameter=2 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=knn_graph +graph_extraction_parameter=6 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=knn_graph +graph_extraction_parameter=6 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=knn_graph +graph_extraction_parameter=8 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=3 +graph_type=knn_graph +graph_extraction_parameter=8 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=eps_graph +graph_extraction_parameter=8 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=eps_graph +graph_extraction_parameter=8 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=eps_graph +graph_extraction_parameter=16 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=eps_graph +graph_extraction_parameter=16 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=eps_graph +graph_extraction_parameter=32 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=eps_graph +graph_extraction_parameter=32 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=knn_graph +graph_extraction_parameter=2 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=knn_graph +graph_extraction_parameter=2 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=knn_graph +graph_extraction_parameter=6 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=knn_graph +graph_extraction_parameter=6 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=knn_graph +graph_extraction_parameter=8 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=5 +graph_type=knn_graph +graph_extraction_parameter=8 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=eps_graph +graph_extraction_parameter=8 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=eps_graph +graph_extraction_parameter=8 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=eps_graph +graph_extraction_parameter=16 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=eps_graph +graph_extraction_parameter=16 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=eps_graph +graph_extraction_parameter=32 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=eps_graph +graph_extraction_parameter=32 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=knn_graph +graph_extraction_parameter=2 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=knn_graph +graph_extraction_parameter=2 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=knn_graph +graph_extraction_parameter=6 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=knn_graph +graph_extraction_parameter=6 
" >> failed_jobs.txt
fi

srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=knn_graph +graph_extraction_parameter=8 

if [ $? -eq 0 ]; then
   echo OK
else
   echo "FAILED srun --cpus-per-task 140 --mem-per-cpu 7G --exclude=bs-hpsvr08 poetry run python experiments/systematic/wl_pc.py +n_iter=10 +graph_type=knn_graph +graph_extraction_parameter=8 
" >> failed_jobs.txt
fi

echo "Done"
