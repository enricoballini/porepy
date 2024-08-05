#!/bin/bash

# rm -r pp.sh.* 
# rm -r launch_mech.sh.*

#PBS -l select=8:ncpus=48:mem=180GB
#PBS -l walltime=00:20:00
#PBS -N EB_job
#PBS -j oe
#PBS -o eb_job.out
#PBS -q echelon_low
# #PBS -q echelon

export OMP_NUM_THREADS=5

echo "Running on nodes: $PBS_NODEFILE" > node_info.txt

python3 main_mech.py > output.txt 2>&1

