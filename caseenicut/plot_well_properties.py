import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import pdb

import read_unrst

os.system("clear")

data_folder_root = "./data"
data_folder = "./data/fluid"
save_folder = "./results/fluid"
os.system("mkdir -p " + save_folder)
timestep = np.loadtxt("./data/TIMESTEP")

training_dataset_id = np.loadtxt(data_folder_root + "/training_dataset_id", dtype=np.int32)
validation_dataset_id = np.loadtxt(data_folder_root + "/validation_dataset_id", dtype=np.int32)
test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id", dtype=np.int32)

unrst_file_name = "case2skew"
variable_names = ["TIME",
                "FGPR", # Field gas production rate
                "FGIR", # field gas injeciton rate
                "WBHP",
                ] # well bottom hole pressure 

fontsize = 28
colors = ["black", "black", "darkorange"]
linestyles = ["-", "--", "-"]


# dates, prop, kw_upper, wg_upper = read_unrst.read_simulated_summary( varsource, filename, init_date)


for idx_mu in test_dataset_id:
    print("writing well properties of " + str(idx_mu))

    for var in variable_names:
        os.system("rm -r " + save_folder + "/" + var + "*")

    read_unrst.read_and_save_variables(data_folder, save_folder, unrst_file_name, idx_mu, variable_names)


variable_names = variable_names[1:] # remove TIME
for idx_mu in test_dataset_id:
    print("plotting well properties of " + str(idx_mu))
    variables = []

    for var_name in variable_names:
        variables.append(np.load(save_folder + "/" + str(idx_mu) +  "/" + var_name + ".npy"))

    # time_final = np.loadtxt("./data/TIMES")[-1]
    # times = np.linspace(0, time_final, variables[0].shape[0])
    times = np.load(save_folder + "/" + str(idx_mu) +  "/TIME.npy" )

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("font", size=fontsize)

    params = {"text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}\usepackage{mathrsfs}"}
    plt.rcParams.update(params)
    matplotlib.rcParams["axes.linewidth"] = 1.5

    fig, ax_1 = plt.subplots(figsize=(12, 4))

    for label in ax_1.get_xticklabels() + ax_1.get_yticklabels():
        label.set_fontsize(fontsize)

    for i, var_name in enumerate(variable_names):
        
        if var_name == "WBHP": ### TODO: improve...
            ax_2 = ax_1.twinx()
            ax_2.plot(
                    times,
                    variables[i] *1e5, # form bar to Pa ### TODO: improve
                    label=var_name,
                    linestyle=linestyles[i],
                    color=colors[i],
                    marker="",
                )
            ax_2.set_ylim(np.array([0, 210e5])) ### upper bound is 1.9 bar, see .DATA TODO: improve 
            ax_2.set_ylabel("$[Pa]$", fontsize=fontsize)
        else:
            ax_1.plot(
                    times,
                    variables[i],
                    label=var_name,
                    linestyle=linestyles[i],
                    color=colors[i],
                    marker="",
                )
            ax_1.set_ylabel(r"$[m^3 / \text{day}]$", fontsize=fontsize)

    plt.savefig(
        save_folder + "/" + str(idx_mu) + "/well_props.pdf",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    # legend:
    handles_all, labels_all = [ (a + b) for a, b in zip( ax_1.get_legend_handles_labels(), ax_2.get_legend_handles_labels() ) ]

    handles = np.ravel(np.reshape(handles_all[:len(variable_names)], (1, len(variable_names))), order="F")
    labels = np.ravel(np.reshape(labels_all[:len(variable_names)], (1, len(variable_names))), order="F")
    fig, ax = plt.subplots(figsize=(25, 10))
    for h, l in zip(handles, labels):
        ax.plot(np.zeros(1), label=l)

    ax.legend(
        handles,
        labels,
        fontsize=fontsize,
        loc="lower center",
        ncol=len(variable_names),
        bbox_to_anchor=(-0.1, -0.65),
    )

    filename = save_folder + "/" + str(idx_mu) + "/well_props_label.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.gcf().clear()

    os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
    os.system("pdfcrop " + filename + " " + filename)




print("\nDone!")