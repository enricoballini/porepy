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

from nnrom.dlromode import offline_ode
import model_fom_case_eni


os.system("clear")

"""
"""
print("\n THIS IS MAIN FLUID \n")

# to try the code:
data_folder_root = "./data"
save_folder_root = "./results"
os.system("mkdir -p " + data_folder_root + "/fluid")
os.system("mkdir -p " + data_folder_root + "/mech")
os.system("mkdir " + save_folder_root)
idx_mu = 99999  # "baseline"

offline = model_fom_case_eni.ModelCaseEni(
    data_folder_root=data_folder_root, save_folder_root=data_folder_root
)
mu_param = np.array([np.log(1e0), np.log(1e0), 1, 5.71e10, 1.0, 1.3e6, 703000.0])
# mu_param = np.array([np.log(1e0), np.log(1e0), 1, 5.71e10, 1.0, 0.0, 0.0]) 
offline.run_one_simulation_no_python(idx_mu, mu_param)

# offline.run_ref_fluid(idx_mu, mu_param)

print("\n\n\n\n\n Part 1 fluid Done!\n\n\n")
stop

#####################################################################################################


# folder preparation:
data_folder_root = "./data"
results_folder_root = "./results"

os.system("mkdir -p " + data_folder_root + "/fluid")
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
num_params = parameters_range.shape[1]

training_dataset_id = np.arange(0, 1)
validation_dataset_id = np.arange(1, 2)
test_dataset_id = np.arange(2, 10)

np.savetxt(data_folder_root + "/test_dataset_id", test_dataset_id)
np.savetxt(results_folder_root + "/test_dataset_id", test_dataset_id)
num_snap_to_generate = test_dataset_id[-1] + 1

# data generation:
model_fom = model_fom_case_eni.ModelCaseEni(data_folder_root, data_folder_root)

offline_data_class = offline_ode.OfflineComputationsODE(
    data_folder=data_folder_root
)
offline_data_class.sample_parameters(
    num_snap_to_generate,
    parameters_range,
    distribution="uniform",
    seed=42,
    rtol=1e-10,
    atol=1e-5,
)

idx_to_generate = np.arange(0, num_snap_to_generate)
# idx_to_generate = np.array([0])


stop

offline_data_class.generate_snapshots_no_python(model_fom, idx_to_generate)

print("\n\n\n\n\n Done!\n\n\n")
stop


# remove extra timesteps:
training_times = np.loadtxt(data_folder + "/TRAINING_TIMES")
ppromode.broomer(
    data_folder,
    training_times,
    np.concatenate((training_dataset_id, validation_dataset_id)),
)

# not sure to what extent I need the foloowing since I interpolate the times when I compute the errors
test_times = np.loadtxt(data_folder + "/TEST_TIMES")
ppromode.broomer(
    data_folder,
    test_times,
    test_dataset_id,
)
