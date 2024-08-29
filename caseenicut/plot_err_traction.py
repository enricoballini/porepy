import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import pdb


sys.path.append("../../mypythonmodulescut")
import nnrom

os.system("clear")


def plt_boxplot_dict(dictionary, save_folder, y_lim_1, y_lim_2, file_name, fontsize=16):
    """ 
    copied from plot_err_cff.py
    """

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("font", size=fontsize)

    params = {"text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}"}
    plt.rcParams.update(params)
    matplotlib.rcParams["axes.linewidth"] = 1.5

    fig, ax_1 = plt.subplots()

    for label in ax_1.get_xticklabels() + ax_1.get_yticklabels():
        label.set_fontsize(fontsize)

    ax_1.boxplot(dictionary.values(), whis=(0, 95), showfliers=True)
    
    ax_1.set_xticklabels(
        [str(int(float(i))) for i in dictionary.keys()] 
    ) 

    ax_1.set_ylim(y_lim_1)
    plt.xlabel("time $[day]$")
    plt.ylabel("Traction error $[N]$")
 
    plt.grid(visible=True, which='major', axis='both', alpha=0.5)

    plt.savefig(
        save_folder + "/" + file_name + ".pdf", dpi=150, bbox_inches="tight", pad_inches=0.2
    )


if __name__ == "__main__":

    data_folder = "./results/nn"
    test_dataset_id = np.loadtxt("./data/test_dataset_id").astype(np.int32)
    # time_file = "./data/TIMES" # for interpolation only
    time_file = "./data/TIMES_MECH" # for interpolation only
    #time_file = "./case_1/data/times_new"
    results_folder_nn = "./results/nn"
    var_names = ["traction"]
    y_lim = 0.5
    x_ticks = np.loadtxt("./data/TIMES_MECH")

    fontsize = 24
    errs_rel_mu_ave = np.loadtxt(data_folder + "/err_relative_mu_ave")
    # errs_abs_mu_ave = np.loadtxt(data_folder + "/err_abs_mu_ave")
    errs_rmse_mu_ave = np.loadtxt(data_folder + "/err_rmse_mu_ave")
    errs_area_mu_ave = np.loadtxt(data_folder + "/err_area_mu_ave")

    errs_rel_mu_ave = np.array([[*errs_rel_mu_ave]])  # sorry, I need two dimensions
    # errs_abs_mu_ave = np.array([[*errs_abs_mu_ave]])
    errs_rmse_mu_ave = np.array([[*errs_rmse_mu_ave]])
    errs_area_mu_ave = np.array([[*errs_area_mu_ave]])

    times = np.loadtxt(time_file)
    
    from nnrom.utils import viz ### usual not understood error

    # viz.plot_err_in_time_relative_and_absolute(
    #     errs_rel_mu_ave,
    #     errs_abs_mu_ave,
    #     times,
    #     var_names,
    #     results_folder_nn,
    #     fontsize,
    #     y_lim,
    #     x_ticks,
    # )

    viz.plot_err_in_time(errs_rmse_mu_ave, times, ["traction_rmse"], results_folder_nn, fontsize)
    viz.plot_err_in_time(errs_area_mu_ave, times, ["traction_area"], results_folder_nn, fontsize)

    err_relative_traction_mu_time = 999999999*np.ones((test_dataset_id.shape[0], times.shape[0]))
    err_area_traction_mu_time = 999999999*np.ones((test_dataset_id.shape[0], times.shape[0]))

    for i, idx_mu in enumerate(test_dataset_id):
        print("loading " + str(idx_mu))
        err_relative_traction_mu_time[i] = np.loadtxt(results_folder_nn + "/" + str(idx_mu) + "/err_relative_vars_time") 
        err_area_traction_mu_time[i] = np.loadtxt(results_folder_nn + "/" + str(idx_mu) + "/err_area_vars_time") 
     
    # realtive: ---------------
    print("plotting relative...")
    dictionary = {
                str(times[1]): err_relative_traction_mu_time[:,1],
                str(times[2]): err_relative_traction_mu_time[:,2],
                }
    y_lim_1 = np.array([0, 0.6])
    y_lim_2 = np.array([0, 0])
    file_name = "boxplot_err_relative_traction"
    plt_boxplot_dict(dictionary, results_folder_nn, y_lim_1, y_lim_2, file_name)


    # realtive area: -----------------
    print("plotting area...")
    dictionary = {
                str(times[1]): err_area_traction_mu_time[:,1],
                str(times[2]): err_area_traction_mu_time[:,2],
                }
    y_lim_1 = np.array([0, 4e4])
    y_lim_2 = np.array([0, 0])
    file_name = "boxplot_err_area_traction"
    plt_boxplot_dict(dictionary, results_folder_nn, y_lim_1, y_lim_2, file_name)
