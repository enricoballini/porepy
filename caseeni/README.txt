

HOW TO RUN: ---------------------------------------------

./run_all.sh
it does everything, from 0 to figures for paper




USEFUL INFO: -----------------------------------------------

to write in real time the output from python code: 
1) python3 -u filename.py
2) inside python code print("...", flush=True)
3) set environment variable PYTHONUNBUFFERED



queue echelon and echelon_low shares the same hardware, made of 
63 nodes, 48 cpus per node, 4gpus per node, 187 GB ram per node
pay attention to the ram, it is less than 187 GB, if you ask for 186GB the job reamins in Q state, without errors. Asking for 180GB the job runs. The actual limit is around 180GB. Yes, the limit declared by qstat -Qf is 180Gb.

echelon_low has a limit of max 32 nodes.

echelon_devel has 4 separates nodes.

qstat
qstat -Qf
qstat -f <job_id>
qstat -u <job_id> 

with the last two commands you can check the nodes pbs selected (exect host), then ssh <node name>

no way to easily see the resources used only by your processes
htop show the cumulative over users usage

at the current time, June 2024, the clusters arent really busy, you can easily steal an entire node for an interactive session with, e.g. qsub -I -X -l select=1:ncpus=24:ngpus=4 -q echelon 

qsub -I -X -l select=1:ncpus=24:ngpus=4 -q echelon        
qsub -I -X -l select=1:ncpus=48:mem=180gb -q echelon_low
export OMP_NUM_THREADS=4


REMEMBER to make sh executable for qsub chmod +x ....sh NO

qselect -u ext2047799 | xargs -r qdel

git does not work on nodi di calcolo, they dont have internet. git work only on login node

watch nvidia-smi 

find ./ -type f -exec sed -i 's/AAA/BBB/g' {} + ### NOT TRIED
grep -rl AAA | xargs sed -i  's/AAA/BBB/g'


UNSOLVED/SOLVED ISSUES: -------------------------------------
- need to code on cluster, connections slow
- SOLVED need to ask cineca to install a recent version of VSCode
- PBS doesn not always retunr errors, you can make some mistake in requireing resources that it automatically fixes them following internal rules
- unpredicatable RAM usage, with 48/5 simulations per node only few failed bcs of memory run out (most simulaitons reached the end), with 48/16 the same... What is the bottleneck?
- partially SOLVED, issues with python multiprocess and multi node job, all the processes go to one one 
- if you ask for too much RAM, PBS put your job in the queue, forever, whitout warnings
- SOLVED No clear documentation of PBS, the one provided by cineca is too simplified, the one on the internet is unclear. Search for altair pbs reference guide version 19.2.1, the one currently installed in summer 2024
- useless laptop, used only to type ssh -X login 
- laptop keyboard without keys...
- IMPORTANT: it seems that subporcess module of python does NOT work on more than one nodes
- i have 300GB of data, how do i manage them? I cant make a safety copy
- pytorch requires specific backends to work with spase matrices on gpu and cpu
- pytorch is unhappy with sparse matrices, no easy way to pass from scipy to torch. *** RuntimeError: add(sparse, dense) is not supported. Use add(dense, sparse) instead.
- Reading/writing from disk (which one?) is terrribly slow, sometimes it takes minutes to just importing the libraries in python!

