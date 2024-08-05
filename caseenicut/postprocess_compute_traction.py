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

for idx_mu in np.concatenate((training_dataset_id, validation_dataset_id, test_dataset_id)):
    print("computing tractions " + str(idx_mu))
    pu.compute_and_save_fault_traction(idx_mu)


print("\nDone!")