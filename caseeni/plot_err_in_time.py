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

    data_folder = "./results/nn"
    test_dataset_id = np.loadtxt("./data/test_dataset_id").astype(np.int32)
    # time_file = "./data/TIMES" # for interpolation only
    time_file = "./data/TIMES_MECH" # for interpolation only
    #time_file = "./case_1/data/times_new"
    save_folder = "./results/nn"
    var_names = ["displacement"]
    y_lim = 0.5
    x_ticks = np.loadtxt("./data/TIMES_MECH")

    fontsize = 24
    errs_rel_mu_ave = np.loadtxt(data_folder + "/err_relative_mu_ave")
    errs_abs_mu_ave = np.loadtxt(data_folder + "/err_absolute_mu_ave")

    errs_rel_mu_ave = np.array([[*errs_rel_mu_ave]])  # sorry, I need two dimensions
    errs_abs_mu_ave = np.array([[*errs_abs_mu_ave]])

    times = np.loadtxt(time_file)
    
    from nnrom.utils import viz # ?

    pdb.set_trace()

    viz.plot_err_in_time_relative_and_absolute(
        errs_rel_mu_ave,
        errs_abs_mu_ave,
        times,
        var_names,
        save_folder,
        fontsize,
        y_lim,
        x_ticks,
    )