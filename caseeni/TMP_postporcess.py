import sys
import os
import pdb
import pickle
import numpy as np
import scipy as sp
import torch
import TMP_postprocess_utils as pu

sys.path.append("../../mypythonmodules")
sys.path.append("../../../mypythonmodules")

import nnrom # weirdly required by torch.load

os.system("clear")



data_folder_root = "./data"

# training_dataset_id = np.loadtxt(data_folder_root + "/training_dataset_id", dtype=np.int32)
# validation_dataset_id = np.loadtxt(data_folder_root + "/validation_dataset_id", dtype=np.int32)
test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id", dtype=np.int32)
test_dataset_id = test_dataset_id[0:2]

nn = torch.load("./resultspartial/nn/model_trained.pth")
nn.set_forward_mode("online")
nn.to("cuda")
nn.eval()

times = np.loadtxt("./data/TIMES_MECH")

for idx_mu in test_dataset_id:
    mu = np.load("./data/mu_param_"+str(idx_mu)+".npy")[[0, 1, 2, 3, 5]]
    for time in times:
        mu_t = torch.tensor([*mu, time], dtype=torch.float32)
        sol_nn = nn(mu_t, "cuda").to("cpu").detach().numpy()
        np.save("./resultspartial/nn/"+str(idx_mu)+"/displacement_"+str(time), sol_nn)

for idx_mu in test_dataset_id:
    print("postprocessing " + str(idx_mu))

    pu.compute_fault_traction(idx_mu)
    pu.compute_fault_traction_nn(idx_mu)

print("\nDone!")