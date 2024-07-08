#!/usr/bin/bash

clear
rm -r run_all.sh.*


# # run fluid: --------------------------------------------------------------------
# # qsub run_fluid.sh
# python3 main_fluid.py

# # wait for fluid to finish: ----------------------------------------------------
# LAST_IDX_MU=$(head -n 1 "./data/last_idx_mu")
# FILE_PATH="./data/fluid/$LAST_IDX_MU/case2skew.ECLEND"
# WAIT_TIME=10 

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
# sleep 600 # to be sure that ALL simulations have finished # TODO: check each folder, not only the last one
# echo "Fluid finished!"



# # run mechanics: -----------------------------------------------------------------
# rm -r "./data/mech/end_file"
# python3 main_mech.py

# # wait for mechanics to finish: -------------------------------------------------------
# LAST_IDX_MU=$(head -n 1 "./data/last_idx_mu")
# WAIT_TIME=60 

# SENTINEL=1
# while [ $SENTINEL -eq 1 ]; do

#   ALL_END_FILES=1 # mech has finished only if EACH folder idx_mu has end_file
#   for i in $(seq 0 $N); do 
#       if [ ! -f "./data/mech/$i/end_file" ]; then
#           ALL_END_FILES=0
#           break
#       fi
#   done
      
#   if [ $ALL_END_FILES -eq 1 ]; then
#     echo "File $FILE_PATH has been generated."
#     SENTINEL=0
#   else
#     echo "Mechanics simulation not finished since file $FILE_PATH has not been found. Checking again in $WAIT_TIME seconds."
#   fi
#   sleep $WAIT_TIME

# done
# sleep 1200
# echo "Mechanics finished!"



# create reduced model: ---------------------------------------------------------------
rm -r "./results/nn/end_file"
rm -r "./data/mech/snap_range"
rm -r "./data/mech/time_range"
qsub run_nn.sh

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
python3 main_nn_part_2.py 
python3 plot_loss.py
python3 plot_err_in_time.py
python3 postprocess.py


echo "\n\n\n\nDone!"
