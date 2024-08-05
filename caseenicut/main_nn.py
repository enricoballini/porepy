import os
import sys
import pdb

import numpy as np
import time

sys.path.append("../../mypythonmodulescut")
sys.path.append("../../../mypythonmodulescut")

import nnrom
from nnrom.dlrom import offline_nn
import model_nn_case_eni


os.system("clear")

print("right after clear")

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

if validation_dataset_id.shape == (): # for debugging
    validation_dataset_id = np.array([validation_dataset_id])

if test_dataset_id.shape == ():
    test_dataset_id = np.array([test_dataset_id])

num_params = (
    np.loadtxt(data_folder_root + "/num_params", dtype=np.int32) + 1
)  # + 1 bcs of time
parameters_range = np.loadtxt(data_folder_root + "/parameters_range")[
    :, [0, 1, 2, 3, 5]
]  # TODO improve it


print("going to copy some files...")
for idx_mu in np.concatenate(
    (training_dataset_id, validation_dataset_id, test_dataset_id)
):
    print(idx_mu)
    os.system(
        "cp ./data/mu_param_"
        + str(idx_mu)
        + ".npy"
        + " ./data/mech/mu_param_"
        + str(idx_mu)
        + ".npy"
    )  # sorry, __getitem__ search for data in only one folder... TODO: decide how to fix this

    os.system(
        "cp ./data/TIMES_MECH "
        + data_folder_mech
        + "/"
        + str(idx_mu)
        + "/PRUNED_TRAINING_TIMES"
    )

    os.system(
        "cp ./data/TIMES_MECH " + data_folder_mech + "/" + str(idx_mu) + "/TIMES_MECH"
    )
    
print("before create datasets")
training_dataset, validation_dataset, test_dataset = offline_nn.create_pytorch_datasets(
    data_folder_mech,
    training_dataset_id,
    validation_dataset_id,
    test_dataset_id,
    var_name="traction_fracture_vector",
)


print("before get min max")
if not os.path.isfile("./data/mech/snap_range"):
    snap_range, time_range = nnrom.utils.misc.get_min_max(
        data_folder_mech, "traction_fracture_vector", time_file="TIMES_MECH"
    )
    np.savetxt("./data/mech/snap_range", snap_range)
    np.savetxt("./data/mech/time_range", time_range)
else:
    print("min max already gotten")
    snap_range = np.loadtxt("./data/mech/snap_range")
    time_range = np.loadtxt("./data/mech/time_range")
    

print("before encoder, decoder, blu")
encoder, decoder, blu = model_nn_case_eni.encoder_decoder_blu(
    data_folder_mech + "/0", num_params
)
print("before scaling matrices")
model_nn_case_eni.count_trainable_params([encoder, decoder, blu], results_folder_nn)
scal_matrices = offline_nn.compute_scaling_matrices(
    snap_range, parameters_range, time_range, scaling_mu_range="01"
)
nn = offline_nn.Dlrom(encoder, decoder, blu, scal_matrices, scaling_mu_range="01")
nn.set_forward_mode("offline")

num_epochs = 1001 #2001 #1001 # 10001
training_batch_size = 32
alpha_1 = 1
alpha_2 = 0.01
alpha_3 = 1
alpha_4 = 0.1
# alpha_traction = None

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

np.savetxt("./results/nn/end_file", np.array([]))

print("\n\n\n\n\nDone!\n\n\n\n")
