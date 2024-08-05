import sys
import os
import pdb
import cProfile
import tracemalloc
import pstats
import io
import time

import numpy as np

"""
"""

sys.path.append("../../mypythonmodulescut")
sys.path.append("../../../mypythonmodulescut")

from nnrom.dlromode import offline_ode
from nnrom.dlrom import offline
import model_fom_case_eni
import read_unrst

os.system("clear")


"""
"""
print("\n THIS IS MAIN MECH, to be run after main_fluid.py \n")

# tracemalloc.start()

# # to try the code:
# data_folder = "./data"
# save_folder = "./results"
# os.system("mkdir -p " + data_folder + "/mech")
# idx_mu = 99999  # "baseline"

# offline = model_fom_case_eni.ModelCaseEni(
#     data_folder_root=data_folder, save_folder_root=data_folder
# )
# mu_param = np.array([np.log(1e0), np.log(1e0), 1, 5.71e10, 1.0, 1.3e6, 703000.0])
# offline.run_one_simulation(idx_mu, mu_param)

# # snapshot = tracemalloc.take_snapshot()
# # top_stats = snapshot.statistics('lineno')
# # print("[ Top 10 ]")
# # with open("./memory.txt", "w") as fle:
# #     for stat in top_stats[:10]:
# #          print(stat)

# print("\n\n\n\n\n Part 1 mech Done!\n\n\n")

# stop

# #####################################################################################################


# folder preparation:
data_folder_root = "./data"
results_folder_root = "./results"

os.system("mkdir -p " + results_folder_root)

# read_unrst.pressure_echelon_to_numpy() # moved in run_all

test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id")
num_snap_to_generate = test_dataset_id[-1] + 1

# data generation:
model_fom = model_fom_case_eni.ModelCaseEni(data_folder_root, data_folder_root)

offline_data_class = offline.OfflineComputations(data_folder_root)

t1 = time.time()
idx_to_generate = np.arange(0, num_snap_to_generate, dtype=np.int32)

print("going to generate snapshots")
offline_data_class.generate_snapshots_no_python_mech(model_fom, idx_to_generate) 

print("\n\n\n\n\n Done!\n\n\n")
