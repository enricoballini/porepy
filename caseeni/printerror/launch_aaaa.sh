#!/usr/bin/bash
echo blablabla
rm -r output_aaaa.txt
rm -r launch_aaaa.sh.*

#PBS -l nodes=1:ncores=2
#PBS -l walltime=24:00:00
#PBS -N aaaa
#PBS -j oe
#PBS -o job_aaaa.out
#PBS -q echelon

export OMP_NUM_THREADS=5
echo "Running on nodes: $PBS_NODEFILE" > node_info.txt

python3 aaaa.py > output_aaaa.txt 2>&1

