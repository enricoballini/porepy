#!/usr/bin/bash
echo $WORK
rm -r pp.sh.* 

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


#PBS -l nodes=1:ppn=10
#PBS -l mem=60gb
#PBS -l walltime=00:30:00
#PBS -N EB_job_pp
# #PBS -q g100_usr_prod
#PBS -e job_err_eb.err
#PBS -o job_out_eb.out
#PBS -q echelon

export OMP_NUM_THREADS=1

python3 main_mech.py > output.txt 2>&1
# python3 sub_model_fom_case_eni.py > output.txt 2>&1




echo $WORK

