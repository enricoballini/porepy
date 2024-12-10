import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import pdb

sys.path.append("../../mypythonmodulescut")
import nnrom

os.system("clear")


if __name__ == "__main__":
    # case_id = int(sys.argv[1])  # 1, 2, 3, ...

    nn_folder = "./results/nn"

    loss_training_list = np.loadtxt(nn_folder + "/loss_ave_list")
    loss_validation_list = np.loadtxt(nn_folder + "/loss_ave_validation_list")

    loss_components_training_list = np.loadtxt(nn_folder + "/loss_components_ave_list")
    loss_components_validation_list = np.loadtxt(
        nn_folder + "/loss_components_ave_validation_list"
    )

    from nnrom.utils import viz

    viz.plot_loss(
        loss_training_list,
        loss_validation_list,
        save_folder=nn_folder,
        validate_every_n_iter=10,  # look at offline_nn
        fontsize=18,
    )

    colors = [
        [0, 0, 0],
        [0.9, 0.1, 0.1],
        [0.2, 0.7, 0.2],
        [0.1, 0.1, 0.9],
        [1, 0.85, 0.55],
    ]
    viz.plot_loss_components(
        loss_components_training_list,
        loss_components_validation_list,
        loss_names=[
            "$\mathscr{L}^1$",
            "$\mathscr{L}^2$",
            "$\mathscr{L}^3$",
            "$\mathscr{L}^4$",
            "$\mathscr{L}^{reg}$",
        ],
        # loss_names = ["$\mathscr{L}_1$", "$\mathscr{L}_2$", "$\mathscr{L}_3$", "$\mathscr{L}_4$", "$\mathscr{L}_5$", "$\mathscr{R}$"],
        save_folder=nn_folder,
        validate_every_n_iter=10,
        fontsize=18,
        colors=colors,
    )
