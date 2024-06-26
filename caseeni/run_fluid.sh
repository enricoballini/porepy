#!/usr/bin/bash

#PBS -l select=1:ncpus=12:ngpus=4
#PBS -l walltime=01:00:00
#PBS -N eb_job_fluid
#PBS -j oe
#PBS -o eb_job_fluid.out
#PBS -q echelon
echo "Running on nodes: $PBS_NODEFILE" > node_info_fluid.txt
python3 main_fluid.py > output_fluid.txt 2>&1
