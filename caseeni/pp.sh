#!/usr/bin/bash
echo $WORK
rm -r pp.sh.* 
#PBS -N job_eb
#PBS -e err_eb.err
#PBS -o out_eb.out # doesnt work
#PBS -j oe
#PBS -l nodes=1
#PBS -l ppn=1 # -l ppn=[count] OR -l mppwidth=[PE_count]
#PBS -q boh  # no errors?
#PBS -l walltime=00:30:00 
echo $WORK
python3 sub_model_fom_case_eni.py > output.txt 2>&1
echo $WORK
