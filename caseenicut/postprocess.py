import sys
import os
import pdb
import pickle
import numpy as np
import scipy as sp
import torch
import postprocess_utils as pu

sys.path.append("../../mypythonmodulescut")
sys.path.append("../../../mypythonmodulescut")

import nnrom # weirdly required by torch.load

os.system("clear")



data_folder_root = "./data"

training_dataset_id = np.loadtxt(data_folder_root + "/training_dataset_id", dtype=np.int32)
validation_dataset_id = np.loadtxt(data_folder_root + "/validation_dataset_id", dtype=np.int32)
test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id", dtype=np.int32)

nn = torch.load("./results/nn/model_trained.pth")
nn.set_forward_mode("online")
nn.to("cuda")
nn.eval()

times = np.loadtxt("./data/TIMES_MECH")

for idx_mu in test_dataset_id:
    os.system("mkdir -p ./results/nn/" + str(idx_mu))
    os.system("mkdir -p ./results/mech/" + str(idx_mu))
    mu = np.load("./data/mu_param_"+str(idx_mu)+".npy")[[0, 1, 2, 3, 5]]
    for time in times:
        mu_t = torch.tensor([*mu, time], dtype=torch.float32)
        sol_nn = nn(mu_t, "cuda").to("cpu").detach().numpy()
        np.save("./results/nn/"+str(idx_mu)+"/traction_fracture_vector_"+str(time), sol_nn)

for idx_mu in test_dataset_id:
    print("postprocessing " + str(idx_mu))

    pu.compute_fault_traction_fom(idx_mu) ### TODO: improve everything...
    pu.compute_fault_traction_nn_cut(idx_mu) ### TODO: improve everything...

print("\nDone!")