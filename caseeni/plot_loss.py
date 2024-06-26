import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import pdb

sys.path.append("../../mypythonmodules")
import nnrom

os.system("clear")


if __name__ == "__main__":
    # case_id = int(sys.argv[1])  # 1, 2, 3, ...

    nn_folder = "./results/nn" 

    loss_training_list = np.loadtxt(nn_folder + "/loss_ave_list")
    loss_validation_list = np.loadtxt(nn_folder + "/loss_ave_validation_list")

    from nnrom.utils import viz
    viz.plot_loss(
        loss_training_list,
        loss_validation_list,
        save_folder=nn_folder,
        validate_every_n_iter=10,  # look at offline_nn
        fontsize=18,
    )