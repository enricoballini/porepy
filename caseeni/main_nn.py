import os
import sys
import pdb

import numpy as np

sys.path.append("../../mypythonmodules")
sys.path.append("../../../mypythonmodules")

import nnrom
from nnrom.dlrom import offline_nn
from nnrom.dlromode import offline_nn_ode
import model_nn_case_eni
import model_fom_case_eni


os.system("clear")

data_folder_root = "./data"
data_folder_mech = "./data/mech"
results_folder_root = "./results"
results_folder_nn = "./results/nn"
os.system("mkdir " + results_folder_nn)

training_dataset_id = np.loadtxt(
    data_folder_root + "/training_dataset_id", dtype=np.int32
)
validation_dataset_id = np.loadtxt(
    data_folder_root + "/validation_dataset_id", dtype=np.int32
)
test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id", dtype=np.int32)

num_params = (
    np.loadtxt(data_folder_root + "/num_params", dtype=np.int32) + 1
)  # + 1 bcs of time
parameters_range = np.loadtxt(data_folder_root + "/parameters_range")[
    :, [0, 1, 2, 3, 5]
]  # TODO improve it


for idx_mu in np.concatenate(
    (training_dataset_id, validation_dataset_id, test_dataset_id)
):
    os.system(
        "cp ./data/mu_param_"
        + str(idx_mu)
        + ".npy"
        + " ./data/mech/mu_param_"
        + str(idx_mu)
        + ".npy"
    )  # sorry, __getitem__ search for data in only one folder... TODO: decide how to fix this

training_dataset, validation_dataset, test_dataset = offline_nn.create_pytorch_datasets(
    data_folder_mech,
    training_dataset_id,
    validation_dataset_id,
    test_dataset_id,
    var_name="displacement",
)

snap_range, time_range = nnrom.utils.misc.get_min_max(
    data_folder_mech, "displacement", time_file="TIMES_MECH"
)
encoder, decoder, blu = model_nn_case_eni.encoder_decoder_blu(
    data_folder_mech + "/0", num_params
)
model_nn_case_eni.count_trainable_params([encoder, decoder, blu], results_folder_nn)
scal_matrices = offline_nn.compute_scaling_matrices(
    snap_range, parameters_range, time_range, scaling_mu_range="01"
)
nn = offline_nn.Dlrom(encoder, decoder, blu, scal_matrices, scaling_mu_range="01")
nn.set_forward_mode("offline")

num_epochs = 21
training_batch_size = 3
alpha_1 = 1
alpha_2 = 1
alpha_3 = 1
alpha_4 = 1

num_updates = num_epochs * training_dataset_id[-1] / training_batch_size
print("I'm going to do a number of weights updates = ", num_updates)
print("check it with number written in NUM_UPDATES_TMP")

offline_nn.train_neural_network(
    results_folder_nn,
    nn,
    training_dataset,
    validation_dataset,
    scal_matrices,
    training_batch_size=training_batch_size,
    epochs=num_epochs,
    lr=1e-3,
    alpha_1=alpha_1,
    alpha_2=alpha_2,
    alpha_3=alpha_3,
    alpha_4=alpha_4,
)


print("\n\n\n before test_trained_neural_network -------------")
np.savetxt("./data/TEST_TIMES", np.loadtxt("./data/TIMES_MECH"))
offline_nn.test_trained_neural_network(
    data_folder_root, data_folder_mech, results_folder_nn, test_dataset, n_eval_pts=4
)


print("\n\n\n before create_vtu_for_figure_nn --------------")
# WHY DONT I SEE nnrom.utils.viz ??
from nnrom.utils import viz
import sub_model_fom_case_eni

viz.create_vtu_for_figure_nn_simply(
    sub_model_fom_case_eni.SubModelCaseEni(),
    data_folder_root,
    results_folder_nn,
    idx_mu_to_plot=test_dataset_id,
)

print("\n\n\n\n\nDone!\n\n\n\n")
