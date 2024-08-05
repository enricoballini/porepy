import os
import sys
import pdb

import numpy as np

sys.path.append("../../mypythonmodulescut")
sys.path.append("../../../mypythonmodulescut")

import nnrom
from nnrom.dlrom import offline_nn
import sub_model_fom_case_eni 


os.system("clear")

data_folder_root = "./data"
data_folder_mech = "./data/mech"
results_folder_root = "./results"
results_folder_nn = "./results/nn"

os.system("mkdir -p " + results_folder_nn)
os.system("mkdir -p ./results/mech")

training_dataset_id = np.loadtxt(
    data_folder_root + "/training_dataset_id", dtype=np.int32
)
validation_dataset_id = np.loadtxt(
    data_folder_root + "/validation_dataset_id", dtype=np.int32
)

test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id", dtype=np.int32)

for idx_mu in test_dataset_id:
    os.system("mkdir -p " + results_folder_nn + "/" + str(idx_mu))

print("before create datasets")
training_dataset, validation_dataset, test_dataset = offline_nn.create_pytorch_datasets(
    data_folder_mech,
    training_dataset_id,
    validation_dataset_id,
    test_dataset_id,
    var_name="traction_fracture_vector",
)

print("before test_trained_neural_network -------------")
np.savetxt("./data/TEST_TIMES", np.loadtxt("./data/TIMES_MECH"))
offline_nn.test_trained_neural_network(
    data_folder_root, data_folder_mech, results_folder_nn, test_dataset, n_eval_pts=4
)

print("before create_vtu_for_figure_nn --------------")
# conceptually wrong if nn approximates qoi      

# # WHY DONT I SEE nnrom.utils.viz ??
# from nnrom.utils import viz

# model_class = sub_model_fom_case_eni.SubModelCaseEni 

# viz.create_vtu_for_figure_nn_simply(
#     model_class,
#     data_folder_root,
#     results_folder_nn,
#     idx_mu_to_plot=test_dataset_id,
# )


np.savetxt("./results/nn/end_file", np.array([]))

print("\n\n\n\n\nDone!")