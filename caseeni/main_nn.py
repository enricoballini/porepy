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
import sub_model_fom_case_eni ###


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

if validation_dataset_id.shape == (): # for debugging
    validation_dataset_id = np.array([validation_dataset_id])

if test_dataset_id.shape == (): # for debugging
    test_dataset_id = np.array([test_dataset_id])

num_params = (
    np.loadtxt(data_folder_root + "/num_params", dtype=np.int32) + 1
)  # + 1 bcs of time
parameters_range = np.loadtxt(data_folder_root + "/parameters_range")[
    :, [0, 1, 2, 3, 5]
]  # TODO improve it


# model = sub_model_fom_case_eni.SubModelCaseEni()### forgot to add some output in the model
# model.subscript = ""###
# model.save_folder = "./CANCELLARE"###
# model.set_geometry()###
# model.set_geometry_part_2()###
# model.set_equation_system_manager()###
# model.create_variables()###
# sd = model.mdg.subdomains(dim=3)[0]###

# volumes_subdomains = np.concatenate(3*[sd.cell_volumes])###
# volumes_interfaces = np.array([])###
# vars_domain = np.array([0])###
# dofs_primary_vars = np.array([np.arange(0, 3*sd.num_cells)])###
# n_dofs_tot = np.array([3*sd.num_cells])###

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
    # np.save(
    #     data_folder_mech + "/" + str(idx_mu) + "/volumes_subdomains", volumes_subdomains
    # ) ###
    # np.save(
    #     data_folder_mech + "/" + str(idx_mu) + "/volumes_interfaces", volumes_interfaces
    # )###
    # np.save(data_folder_mech + "/" + str(idx_mu) + "/vars_domain", vars_domain)###
    # np.save(data_folder_mech + "/" + str(idx_mu) + "/dofs_primary_vars", dofs_primary_vars) ###
    # np.save(data_folder_mech + "/" + str(idx_mu) + "/n_dofs_tot", n_dofs_tot)###


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

num_epochs = 501
training_batch_size = 4
alpha_1 = 1
alpha_2 = 0.01
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

model_class = sub_model_fom_case_eni.SubModelCaseEni ### forgot to add some output in the model # no...

viz.create_vtu_for_figure_nn_simply(
    model_class,
    data_folder_root,
    results_folder_nn,
    idx_mu_to_plot=test_dataset_id,
)


np.savetxt("./data/mech/end_file", np.array([]))
print("\n\n\n\n\nDone!\n\n\n\n")
