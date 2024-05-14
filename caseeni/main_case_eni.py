import sys
import os
import pdb
import cProfile
import pstats
import io

import numpy as np

"""
torc>>> torch.__version__
'1.13.1+cu117'
>>> import torchvision
>>> torchvision.__version__
'0.14.1+cu117'
>>> import torchaudio
>>> torchaudio.__version__
'0.13.1+cu117'

pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
"""

sys.path.remove("/home/inspiron/Desktop/PhD/porepy/src")
sys.path.append("/home/inspiron/Desktop/PhD/pprom")
sys.path.append("/home/inspiron/Desktop/PhD/eni_venv/porepy/src")

import torch

# import ppromode
import model_fom_case_eni
import model_nn_case_eni


os.system("clear")


"""
"""

data_folder = "./data"
idx_mu = 000
save_folder = "./results" + str(idx_mu)
model = model_fom_case_eni.ModelCaseEni(data_folder=data_folder)
mu_param = np.array([0.2])
model.solve_one_instance_ti_tf(mu_param, save_folder, idx_mu)

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

# settings:
parameters_range = np.array([[np.log(5e-3), -0.15], [np.log(2e3), 0.05]])
num_params = parameters_range.shape[1]

training_dataset_id = np.arange(0, 100)
validation_dataset_id = np.arange(100, 120)
test_dataset_id = np.arange(120, 140)

np.savetxt(data_folder + "/test_dataset_id", test_dataset_id)
np.savetxt(results_folder + "/test_dataset_id", test_dataset_id)
num_snap_to_generate = test_dataset_id[-1] + 1

# data generation:
model_fom = model_case_4.ModelCase4(data_folder)

offline_data_class = ppromode.offline_ode.OfflineComputationsODE(data_folder)
offline_data_class.sample_parameters(
    num_snap_to_generate,
    parameters_range,
    distribution="uniform",
    seed=42,
    rtol=1e-10,
    atol=1e-5,
)

t_0 = 1e-2
timestep = 1e-4 / t_0
timestep_nn = 8 * timestep
time_final_training = 160 * timestep
time_final_test = 200 * timestep

np.savetxt(data_folder + "/TIMESTEP", np.array([timestep]))
np.savetxt(data_folder + "/TIMESTEP_NN", np.array([timestep_nn]))
np.savetxt(
    data_folder + "/TRAINING_TIMES",
    np.arange(0, time_final_training + timestep_nn, timestep_nn),
)
np.savetxt(
    data_folder + "/TEST_TIMES",
    np.arange(0, time_final_test + timestep_nn, timestep_nn),
)

# offline_data_class.generate_snapshots(
#     model_fom, np.arange(0, num_snap_to_generate), n_proc=6
# )


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
    ppromode.offline_nn_ode.create_pytorch_datasets(
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
scal_matrices = ppromode.offline_nn_ode.compute_scaling_matrices(
    snap_range, parameters_range, time_range, scaling_mu_range="01"
)
nn = ppromode.offline_nn_ode.DlromODE(
    data_folder, encoder, decoder, blu, scal_matrices, scaling_mu_range="01"
)
nn.set_forward_mode("offline")

# pr = cProfile.Profile()
# pr.enable()
num_epochs = 1
training_batch_size = 1
ppromode.offline_nn_ode.train_neural_network(
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
ppromode.offline_nn_ode.test_trained_neural_network(
    data_folder, results_folder, test_dataset, n_eval_pts=100
)


print("\n\n\n before create_vtu_for_figure_nn --------------")
ppromode.viz_ode.create_vtu_for_figure_nn(
    model_fom, data_folder, results_folder, idx_mu_to_plot=test_dataset_id
)

print("\nDone!")
