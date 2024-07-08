#!/usr/bin/bash

#PBS -l select=1:ncpus=12:ngpus=4
#PBS -l walltime=96:00:00
#PBS -N eb_job_nn
#PBS -j oe
#PBS -o eb_job_nn.out
#PBS -q echelon

echo "Running on nodes: $PBS_NODEFILE" > node_info_nn.txt
python3 -u main_nn.py > output_nn.txt 2>&1
