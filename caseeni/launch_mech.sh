#!/usr/bin/bash
rm -r pp.sh.* 
rm -r launch_mech.sh.*

# #PBS -N job_eb
# #PBS -e err_eb.err
# #PBS -o out_eb.out # doesnt work
# #PBS -j oe
# #PBS -l nodes=1
# #PBS -l ppn=1 # -l ppn=[count] OR -l mppwidth=[PE_count]
# #PBS -q echelon  ### 
# #PBS -l walltime=00:30:00 
# echo $WORK
# python3 sub_model_fom_case_eni.py > output.txt 2>&1



#!/bin/bash

# #PBS -N GAN
# #PBS -l select=1:ncpus=48:ngpus=4
# #PBS -l walltime=40:00:00       
# #PBS -q echelon_low
# #PBS -j oe                      
# #PBS -o test_gan.out    

# cd "${PBS_O_WORKDIR}"

# module purge
# module load autoload venvpython_flumi_gan/3.8

# echo "===> python is $(which python)"
# echo "===> modules ===="
# module list
# python conditioning_model_3d.py &> myoutput_gan.out



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

