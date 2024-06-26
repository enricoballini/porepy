#!/usr/bin/bash
rm -r pp.sh.* 
rm -r launch_mech.sh.*

# #!/bin/bash

#PBS -l nodes=1:ncores=48 # ppn=10
#   #PBS -l mem=60gb
#PBS -l walltime=24:00:00
#PBS -N EB_job_pp
#PBS -j oe
#PBS -o job_out_eb.out
# #PBS -q echelon_low
#PBS -q echelon

export OMP_NUM_THREADS=5

echo "Running on nodes: $PBS_NODEFILE" > node_info.txt

python3 main_mech.py > output.txt 2>&1

