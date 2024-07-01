

HOW TO RUN: ---------------------------------------------

./run_all.sh
it does everything, from 0 to figures for paper




USEFUL INFO: -----------------------------------------------

to write in real time the output from python code: 
1) python3 -u filename.py
2) inside python code print("...", flush=True)
3) set environment variable PYTHONUNBUFFERED



queue echelon and echelon_low shares the same hardware, made of 
63 nodes, 48 cpus per node, 187 GB ram per node
echelon_low has a limit of max 32 nodes.

echelon_devel has 4 separates nodes.


qstat
qstat -Qf
qstat -f <job_id>
qstat -u <job_id> 

with the last two commands you can check the nodes pbs selected (exect host), then ssh <node name>

no way to easily see the resources used only by your processes
htop show the cumulative over users usage



 ---------------------------------------

