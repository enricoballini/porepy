import sys
import os
import pdb
import pickle
import numpy as np
import scipy as sp
import torch
import postprocess_utils as pu
import pdb

sys.path.append("../../mypythonmodulescut")
sys.path.append("../../../mypythonmodulescut")

os.system("clear")


"""
find idx_mu and time of the local peack of cff. Actually sort all the peacks and save them
"""


data_folder_root = "./data"
results_folder_root = "./results"

training_dataset_id = np.loadtxt(data_folder_root + "/training_dataset_id", dtype=np.int32)
validation_dataset_id = np.loadtxt(data_folder_root + "/validation_dataset_id", dtype=np.int32)
test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id", dtype=np.int32)

times = np.loadtxt("./data/TIMES_MECH")

cff_maxes = np.zeros([test_dataset_id.shape[0], times.shape[0], 2])

for ii, idx_mu in enumerate(test_dataset_id):
    for jj, time in enumerate(times):
        cff_maxes[ii] = np.load(results_folder_root + "/mech/" + str(idx_mu) + "/cff.npy")
        cff_maxes[ii,jj,:] = [idx_mu, time]


pdb.set_trace()

cff_maxes_sorted = np.sort(
    cff_maxes.reshape((test_dataset_id.shape[0]*times.shape[0], 2), order="F"), axis=0
) # o  C o F...


print(cff_maxes_sorted)
np.savetxt(results_folder_root + "/mech/cff_maxes_sorted", cff_maxes_sorted)

print("\nDone!")

