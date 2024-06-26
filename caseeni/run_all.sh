#!/usr/bin/bash

clear
rm -r run_all.sh.*


# run fluid: --------------------------------------------------------------------
qsub run_fluid.sh

# wait for fluid to finish: ----------------------------------------------------
LAST_IDX_MU=$(head -n 1 "./data/last_idx_mu")
FILE_PATH="./data/fluid/$LAST_IDX_MU/case2skew.ECLEND"
WAIT_TIME=10 

SENTINEL=1
while [ $SENTINEL -eq 1 ]; do
  if [ -f "$FILE_PATH" ]; then
    echo "File $FILE_PATH has been generated."
    SENTINEL=0
  else
    echo "Fluid simulation not finished since file $FILE_PATH has not been found. Checking again in $WAIT_TIME seconds."
  fi
  sleep $WAIT_TIME
done
echo "Fluid finished!"



# run mechanics: -----------------------------------------------------------------
qsub run_mech.sh

# wait for mechanics to finish: -------------------------------------------------------
FILE_PATH="./data/mech/end_file"
WAIT_TIME=60 

SENTINEL=1
while [ $SENTINEL -eq 1 ]; do
  if [ -f "$FILE_PATH" ]; then
    echo "File $FILE_PATH has been generated."
    SENTINEL=0
  else
    echo "Mechanics simulation not finished since file $FILE_PATH has not been found. Checking again in $WAIT_TIME seconds."
  fi
  sleep $WAIT_TIME
done
echo "Mechanics finished!"



# create reduced model: ---------------------------------------------------------------
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
python3 plot_loss.py
python3 plot_err_in_time.py
python3 post_process.py


echo "\n\n\n\nDone!"





