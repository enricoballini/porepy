import os
import sys
import pdb

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

os.system("clear")

data_folder_root = "./data"
results_folder_mech = "./results/mech"
results_folder_nn = "./results/nn"

training_dataset_id = np.loadtxt(data_folder_root + "/training_dataset_id", dtype=np.int32)
validation_dataset_id = np.loadtxt(data_folder_root + "/validation_dataset_id", dtype=np.int32)
test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id", dtype=np.int32)

times_mech = np.loadtxt("./data/TIMES_MECH")



def plt_boxplot_dict(dictionary, save_folder, y_lim_1, y_lim_2, file_name, fontsize=16):
    """ 
    copied from viz
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

    # for label in ax_2.get_xticklabels() + ax_2.get_yticklabels():
    #     label.set_fontsize(fontsize)

    ax_1.boxplot(dictionary.values(), whis=(0, 97), showfliers=True)
    # ax_2.boxplot(dictionary.values(), showfliers=False)

    ax_1.set_xticklabels(
        dictionary.keys()
    )  # I'm  writing the labels twice
    # plt.yscale("log")

    ax_1.set_ylim(y_lim_1)
    # ax_2.set_ylim(y_lim_2)

    # ax_1.spines["bottom"].set_visible(False)
    # ax_2.spines["top"].set_visible(False)

    # ax_1.xaxis.tick_top()
    # ax_1.tick_params(labeltop=False)  # don't put tick labels at the top
    # ax_2.xaxis.tick_bottom()

    # d = 0.015  #
    # kwargs = dict(transform=ax_1.transAxes, color="k", clip_on=False)
    # ax_1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    # ax_1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    # kwargs.update(transform=ax_2.transAxes)  # switch to the bottom axes
    # ax_2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # ax_2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plt.savefig(
        save_folder + "/" + file_name + ".pdf", dpi=150, bbox_inches="tight", pad_inches=0.2
    )


err_relative_cff_mu_time = 999999999*np.ones((test_dataset_id.shape[0], times_mech.shape[0]))
err_area_cff_mu_time = 999999999*np.ones((test_dataset_id.shape[0], times_mech.shape[0]))
err_relative_vs_max_cff_mu_time = 999999999*np.ones((test_dataset_id.shape[0], times_mech.shape[0]))

for i, idx_mu in enumerate(test_dataset_id):
    print("loading " + str(idx_mu))
    err_relative_cff_mu_time[i] = np.loadtxt(results_folder_nn + "/" + str(idx_mu) + "/err_relative_cff") 
    err_area_cff_mu_time[i] = np.loadtxt(results_folder_nn + "/" + str(idx_mu) + "/err_area_cff") 
    err_relative_vs_max_cff_mu_time[i] = np.loadtxt(results_folder_nn + "/" + str(idx_mu) + "/err_relative_vs_max_cff") 


# realtive: ---------------
print("plotting relative...")
dictionary = {
            str(times_mech[1]): err_relative_cff_mu_time[:,1],
            str(times_mech[2]): err_relative_cff_mu_time[:,2],
            }
y_lim_1 = np.array([0, 0.6])
y_lim_2 = np.array([0, 0])
file_name = "boxplot_err_relative_cff"
plt_boxplot_dict(dictionary, results_folder_nn, y_lim_1, y_lim_2, file_name)


# area: -----------------
print("plotting_area...")
dictionary = {
            str(times_mech[1]): err_area_cff_mu_time[:,1],
            str(times_mech[2]): err_area_cff_mu_time[:,2],
            }
y_lim_1 = np.array([0, 1e5])
y_lim_2 = np.array([0, 0])
file_name = "boxplot_err_area_cff"
plt_boxplot_dict(dictionary, results_folder_nn, y_lim_1, y_lim_2, file_name)

# relative vs max: ----------------------
print("plotting relative vs max...")
dictionary = {
            str(times_mech[1]): err_relative_vs_max_cff_mu_time[:,1],
            str(times_mech[2]): err_relative_vs_max_cff_mu_time[:,2],
            }
y_lim_1 = np.array([0, 0.3])
y_lim_2 = np.array([0, 0])
file_name = "boxplot_err_relative_vs_max_cff"
plt_boxplot_dict(dictionary, results_folder_nn, y_lim_1, y_lim_2, file_name)






print("\nDone!")
