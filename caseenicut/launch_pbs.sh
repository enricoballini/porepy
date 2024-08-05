#!/usr/bin/bash

# #PBS -S /usr/bin/bash

echo $WORK

# this should be enough:
# #PBS -l nodes=1
# #PBS -l ppn=1 # -l ppn=[count] OR -l mppwidth=[PE_count]
# #PBS -q boh
# #PBS -l walltime=00:30:00 

#PBS -N job_eb
#PBS -e err_eb.err
#PBS -o out_eb.out
#PBS -j oe


# #PBS -l mppnppn 1 # = #SBATCH--ntasks-per-node=1
# #PBS -l mem= 80000# in MB 


# from user guide:
# #PBS -N job_eb
# #PBS -q <QUEUE>
# #PBS -e job_err_eb.err
# #PBS -o job_out_eb.out
# #PBS -l select=1:ncpus=1

echo $WORK
export OMP_NUM_THREADS=1

#srun --cpu-bind=cores python3 main_mech.py > output.txt 2>&1
srun --cpu-bind=cores python3 sub_model_fom_case_eni.py > output.txt 2>&1

echo $WORK
