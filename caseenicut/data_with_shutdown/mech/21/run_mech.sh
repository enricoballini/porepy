#!/usr/bin/bash

#PBS -l select=1:ncpus=5:mem=18GB
#PBS -l walltime=96:00:00
#PBS -N eb_job_mech
#PBS -j oe
#PBS -o eb_job_mech.out
#PBS -q echelon_low

export OMP_NUM_THREADS=5
echo "Running on nodes: $PBS_NODEFILE" > node_info_mech.txt
cd ./caseenicut/data/mech/21
python3 -u mech.py > output_mech.txt 2>&1
            