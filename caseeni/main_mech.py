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

sys.path.append("../../mypythonmodules")

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

os.system("mkdir -p " + data_folder_root + "/mech")
os.system("rm -r " + results_folder_root)
os.system("mkdir -p " + results_folder_root)

read_unrst.pressure_echelon_to_numpy()


test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id")
num_snap_to_generate = test_dataset_id[-1] + 1

# data generation:
model_fom = model_fom_case_eni.ModelCaseEni(data_folder_root, data_folder_root)

offline_data_class = offline.OfflineComputations(data_folder_root)

t1 = time.time()
idx_to_generate = np.arange(0, num_snap_to_generate, dtype=np.int32)
# idx_to_generate = np.arange(0, 10)
offline_data_class.generate_snapshots(model_fom, idx_to_generate, n_proc=23)
print("\nTOTAL TIME = ", time.time() - t1)


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


# train reduced model:
os.system("rm -r " + data_folder + "/99991")
os.system("rm -r " + data_folder + "/99992")
os.system("rm -r " + data_folder + "/99993")
os.system("rm -r " + data_folder + "/99994")
training_dataset, validation_dataset, test_dataset = (
    offline_nn_ode.create_pytorch_datasets(
        data_folder,
        training_dataset_id,
        validation_dataset_id,
        test_dataset_id,
    )
)


snap_range, time_range = ppromode.misc_ode.get_min_max(data_folder, "solution")
encoder, decoder, blu = model_nn_case_4.encoder_decoder_blu(
    data_folder + "/0", num_params
)  # I search for the data in the first folder
scal_matrices = offline_nn_ode.compute_scaling_matrices(
    snap_range, parameters_range, time_range, scaling_mu_range="01"
)
nn = offline_nn_ode.DlromODE(
    data_folder, encoder, decoder, blu, scal_matrices, scaling_mu_range="01"
)
nn.set_forward_mode("offline")

# pr = cProfile.Profile()
# pr.enable()
num_epochs = 1
training_batch_size = 1
offline_nn_ode.train_neural_network(
    data_folder,
    results_folder,
    nn,
    training_dataset,
    validation_dataset,
    scal_matrices,
    training_batch_size=training_batch_size,
    epochs=num_epochs,  #
    lr=1e-3,
    alpha_1=alpha_1,
    alpha_2=alpha_2,
    alpha_3=alpha_3,
    alpha_4=alpha_4,
    alpha_5=alpha_5,
)
# pr.disable()
# s = io.StringIO()
# sortby = pstats.SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())


num_updates = num_epochs * training_dataset_id[-1] / training_batch_size
print("I'm going to do a number of weights updates = ", num_updates)
print("check it with number written in NUM_UPDATES_TMP")

print("\n\n\n before test_trained_neural_network -------------")
offline_nn_ode.test_trained_neural_network(
    data_folder, results_folder, test_dataset, n_eval_pts=100
)


print("\n\n\n before create_vtu_for_figure_nn --------------")
ppromode.viz_ode.create_vtu_for_figure_nn(
    model_fom, data_folder, results_folder, idx_mu_to_plot=test_dataset_id
)

print("\nDone!")
