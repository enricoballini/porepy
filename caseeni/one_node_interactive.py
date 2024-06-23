qsub -I -X -l select=1:ncpus=24:ngpus=4 -q echelon        
qsub -I -X -l select=1:ncpus=48:mem=180gb -q echelon_low
export OMP_NUM_THREADS=4
