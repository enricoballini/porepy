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

# sys.path.append("/home/inspiron/Desktop/PhD/mypythonmodules")
# sys.path.append("/g100_work/pMI24_MatBa/eballin1/mypythonmodules")  # Cineca G100
sys.path.append("../../mypythonmodules")  # hpc5 (and others for the future)


# from pprom import ppromode
from ppromode import offline_ode
import model_fom_case_eni
import model_nn_case_eni


os.system("clear")


"""
"""

data_folder = "./data"
save_folder = "./results"
os.system("mkdir -p " + data_folder + "/fliud")
os.system("mkdir -p " + data_folder + "/mech")
os.system("mkdir " + save_folder)
idx_mu = 99999

time_final_training = 40 * 365.25
timestep = time_final_training / 40
np.savetxt(data_folder + "/TIMESTEP", np.array([timestep]))
np.savetxt(
    data_folder + "/TIMES",
    np.arange(0, time_final_training + timestep, timestep),
)
offline = model_fom_case_eni.ModelCaseEni(data_folder=data_folder)
mu_param = np.array([np.log(1e5), np.log(1e-1), 5.71e10, 5.71e10, 1.0, 1.5e6, 703000.0])
# offline.run_one_simulation(data_folder, save_folder, idx_mu, mu_param)
offline.run_one_simulation_no_python(data_folder, data_folder, idx_mu, mu_param)

print("\n\n\n\n\n Part 1 Done!\n\n\n")
stop

#####################################################################################################

alpha_1 = 1
alpha_2 = 1
alpha_3 = 1
alpha_4 = 1
alpha_5 = 1


# folder preparation:
data_folder = "./data"
results_folder = "./results"

os.system("mkdir -p " + data_folder + "/fluid")
os.system("mkdir -p " + data_folder + "/mech")
os.system("rm -r " + results_folder)
os.system("mkdir -p " + results_folder)


# settings:
#  0    1           2                           3               4       5               6
# Ka, K_\perp, Yung modulus outside, Young's modulus reservoir, Cm, injection_rate, production_rate
parameters_range = np.array(
    [
        [np.log(1e-2), np.log(1e-6), 1e9, 1e9, 1.0, 1.0e6, 2 * 703000.0],
        [np.log(1e5), np.log(1e-1), 5.71e10, 5.71e10, 1.0, 1.5e6, 2 * 703000.0],
    ]
)
num_params = parameters_range.shape[1]

training_dataset_id = np.arange(0, 1)
validation_dataset_id = np.arange(1, 2)
test_dataset_id = np.arange(2, 46)

np.savetxt(data_folder + "/test_dataset_id", test_dataset_id)
np.savetxt(results_folder + "/test_dataset_id", test_dataset_id)
num_snap_to_generate = test_dataset_id[-1] + 1

# data generation:
model_fom = model_fom_case_eni.ModelCaseEni(data_folder)

offline_data_class = offline_ode.OfflineComputationsODE(data_folder)
offline_data_class.sample_parameters(
    num_snap_to_generate,
    parameters_range,
    distribution="uniform",
    seed=42,
    rtol=1e-10,
    atol=1e-5,
)

model_fom.run_one_simulation_no_python(
    np.load(data_folder + "/mu_param_0.npy"), data_folder, idx_mu=0
)


t1 = time.time()
idx_to_generate = np.arange(0, num_snap_to_generate)
# idx_to_generate = np.array([0])

offline_data_class.generate_snapshots(model_fom, idx_to_generate, n_proc=6)
print("\nTOTAL TIME = ", time.time() - t1)


print("\n\n\n\n\n Part 1 Done!\n\n\n")
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
