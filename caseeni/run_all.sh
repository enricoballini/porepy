#!/usr/bin/bash

clear
rm -r run_all.sh.*


# # run fluid: --------------------------------------------------------------------
# #PBS -l nodes=1:ncores=12:ngpus=4
# #PBS -l walltime=01:00:00
# #PBS -N eb_job_fluid
# #PBS -j oe
# #PBS -o eb_job_fluid.out
# #PBS -q echelon
# echo "Running on nodes: $PBS_NODEFILE" > node_info_fluid.txt
# python3 main_fluid.py > output_fluid.txt 2>&1

# # wait for fluid to finish: ----------------------------------------------------
# LAST_IDX_MU=$(head -n 1 "./data/last_idx_mu")
# FILE_PATH="./data/fluid/$LAST_IDX_MU/case2skew.ECLEND"
# WAIT_TIME=5 

# SENTINEL=1
# while [ $SENTINEL -eq 1 ]; do
#   if [ -f "$FILE_PATH" ]; then
#     echo "File $FILE_PATH has been generated."
#     SENTINEL=0
#   else
#     echo "Fluid simulation not finished since file $FILE_PATH has not been found. Checking again in $WAIT_TIME seconds."
#   fi
#   sleep $WAIT_TIME
# done
# echo "Fluid finished!"



# # run mechanics: -----------------------------------------------------------------
# #PBS -l nodes=1:ncores=48 # ppn=10
# #   #PBS -l mem=60gb
# #PBS -l walltime=96:00:00
# #PBS -N eb_job_mech
# #PBS -j oe
# #PBS -o eb_job_mech.out
# # #PBS -q echelon_low
# #PBS -q echelon

# export OMP_NUM_THREADS=8 ###
# echo "Running on nodes: $PBS_NODEFILE" > node_info_mech.txt
# python3 main_mech.py > output_mech.txt 2>&1


# # wait for mechanics to finish: -------------------------------------------------------
# FILE_PATH="./data/mech/end_file"
# WAIT_TIME=60 

# SENTINEL=1
# while [ $SENTINEL -eq 1 ]; do
#   if [ -f "$FILE_PATH" ]; then
#     echo "File $FILE_PATH has been generated."
#     SENTINEL=0
#   else
#     echo "Mechanics simulation not finished since file $FILE_PATH has not been found. Checking again in $WAIT_TIME seconds."
#   fi
#   sleep $WAIT_TIME
# done
# echo "Mechanics finished!"



# create reduced model: ---------------------------------------------------------------
#PBS -l nodes=1:ncores=12:ngpus=4
#PBS -l walltime=01:00:00
#PBS -N eb_job_nn
#PBS -j oe
#PBS -o eb_job_nn.out
#PBS -q echelon
echo "Running on nodes: $PBS_NODEFILE" > node_info_nn.txt
python3 main_nn.py > output_nn.txt 2>&1

# wait for ROM to finish: ----------------------------------------------------
LAST_IDX_MU=$(head -n 1 "./data/last_idx_mu")
FILE_PATH="./results/nn/end_file"
WAIT_TIME=10 

SENTINEL=1
while [ $SENTINEL -eq 1 ]; do
  if [ -f "$FILE_PATH" ]; then
    echo "File $FILE_PATH has been generated."
    SENTINEL=0
  else
    echo "Reduced model computations not finished since file $FILE_PATH has not been found. Checking again in $WAIT_TIME seconds."
  fi
  sleep $WAIT_TIME
done
echo "ROM finished!"


# figures: ----------------------------------------------------------------------------
python3 plot_loss.py
python3 plot_err_in_time.py



echo "Done!"





