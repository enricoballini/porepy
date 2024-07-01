#!/usr/bin/bash

#PBS -l select=40:ncpus=48:mem=186GB
#PBS -l walltime=96:00:00
#PBS -N eb_job_mech
#PBS -j oe
#PBS -o eb_job_mech.out
#PBS -q echelon

export OMP_NUM_THREADS=8
echo "Running on nodes: $PBS_NODEFILE" > node_info_mech.txt
python3 -u main_mech.py > output_mech.txt 2>&1
