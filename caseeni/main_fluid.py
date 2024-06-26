import sys
import os
import pdb
import cProfile
import pstats
import io
import time

import numpy as np

"""
https://pytorch.org/get-started/previous-versions/
"""

sys.path.append("../../mypythonmodules")
sys.path.append("../../../mypythonmodules")

from nnrom.dlromode import offline_ode
from nnrom.dlrom import offline
import model_fom_case_eni


os.system("clear")

"""
"""
print("\n THIS IS MAIN FLUID \n")

# # to try the code:
# data_folder_root = "./data"
# save_folder_root = "./results"
# os.system("mkdir -p " + data_folder_root + "/fluid")
# os.system("mkdir -p " + data_folder_root + "/mech")
# os.system("mkdir " + save_folder_root)
# idx_mu = 99999  # "baseline"

# offline = model_fom_case_eni.ModelCaseEni(
#     data_folder_root=data_folder_root, save_folder_root=data_folder_root
# )
# mu_param = np.array([np.log(1e0), np.log(1e0), 1, 5.71e10, 1.0, 1.3e6, 703000.0])
# # mu_param = np.array([np.log(1e0), np.log(1e0), 1, 5.71e10, 1.0, 0.0, 0.0])
# offline.run_one_simulation_no_python(idx_mu, mu_param)

# # offline.run_ref_fluid(idx_mu, mu_param)

# print("\n\n\n\n\n Part 1 fluid Done!\n\n\n")
# stop

#####################################################################################################


# folder preparation:
data_folder_root = "./data"
results_folder_root = "./results"

os.system("mkdir -p " + data_folder_root + "/fluid")
os.system("mkdir -p " + data_folder_root + "/mech")
os.system("rm -r " + results_folder_root)
os.system("mkdir -p " + results_folder_root)


# settings:
#  0                 1              2                           3               4       5               6
# Ka multiplier, K_\perp mult, Yung modulus ratio, Young's modulus reservoir, Cm, injection_rate, production_rate
# ech,              ech,            pp,                     pp,             , ech,           ech,        ech,
parameters_range = np.array(
    [
        [np.log(1e-1), np.log(1e-6), 1, 5.71e9, 1.0, 1.0e6, 703000.0],
        [np.log(1e2), np.log(1e1), 25, 5.71e11, 1.0, 1.5e6, 703000.0],
    ]
)
np.savetxt(data_folder_root + "/parameters_range", parameters_range)
num_params = parameters_range.shape[1] - 2  # Cm and production rate aren't
np.savetxt(data_folder_root + "/num_params", np.array([num_params]), fmt="%d")


training_dataset_id = np.arange(0, 700, dtype=np.int32)
validation_dataset_id = np.arange(700, 800, dtype=np.int32)
test_dataset_id = np.arange(800, 900, dtype=np.int32)

# training_dataset_id = np.arange(0, 3, dtype=np.int32) ###
# validation_dataset_id = np.arange(3, 4, dtype=np.int32) ###
# test_dataset_id = np.arange(4, 6, dtype=np.int32) ###


np.savetxt(data_folder_root + "/training_dataset_id", training_dataset_id, fmt="%d")
np.savetxt(data_folder_root + "/validation_dataset_id", validation_dataset_id, fmt="%d")
np.savetxt(data_folder_root + "/test_dataset_id", test_dataset_id, fmt="%d")
np.savetxt(results_folder_root + "/test_dataset_id", test_dataset_id, fmt="%d")
num_snap_to_generate = test_dataset_id[-1] + 1

with open("./data/last_idx_mu", "+w") as fle:
    fle.write(str(num_snap_to_generate-1))
# np.savetxt("./data/last_idx_mu", np.array([num_snap_to_generate], dtype=np.int32)) # read by run_all.sh

# data generation:
model_fom = model_fom_case_eni.ModelCaseEni(data_folder_root, data_folder_root)

offline_data_class = offline.OfflineComputations(data_folder=data_folder_root)
offline_data_class.sample_parameters(
    num_snap_to_generate,
    parameters_range,
    distribution="latin",
    seed=42,
    rtol=1e-10,
    atol=1e-5,
)


idx_to_generate = np.arange(0, num_snap_to_generate)
# idx_to_generate = np.array([0])

offline_data_class.generate_snapshots_no_python(model_fom, idx_to_generate)

print("\n\n\n\n\n Fluid done!\n\n\n")
