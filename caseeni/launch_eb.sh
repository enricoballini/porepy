#!/bin/bash
# for slurm cpu = thread, https://slurm.schedmd.com/cpu_management.html

#SBATCH --nodes=1
# #SBATCH --ntasks-per-socket= # Controls the maximum number of tasks per allocated socket
#SBATCH --ntasks-per-node=1
# #SBATCH --ntasks-per-core=1 # Controls the maximum number of tasks per allocated core
#SBATCH --cpus-per-task=1
# #SBATCH --threads-per-core=1
# #SBATCH --ntasks = # Controls the number of tasks to be created for the job

# #SBATCH --cpu-bind=cores # Automatically generate masks binding tasks to cores/threads/....  https://slurm.schedmd.com/cpu_management.html and https://slurm.schedmd.com/srun.html#OPT_cpu-bind	 
	 
#SBATCH --mem=60GB # default is 7800MB per core https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.3%3A+GALILEO100+UserGuide

#SBATCH --time=00:30:00
#SBATCH --account=pMI24_MatBa
#SBATCH --partition=g100_usr_prod
#SBATCH --job-name=EB_job_pp
#SBATCH --err=job_err_eb.err
#SBATCH --out=job_out_eb.out



export OMP_NUM_THREADS=1

srun --cpu-bind=cores python3 main_mech.py > output.txt 2>&1

