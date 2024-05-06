import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import pdb


os.system("clear")


def load_output_mass(filename):
    """ """
    data = np.loadtxt(filename, delimiter=",")
    return (
        data[:, 0],  # time
        data[:, 1],  # mass
    )


if __name__ == "__main__":
    case_id = int(sys.argv[1])  # 1, 2, 3

    #####################################################

    if case_id == 1:
        case_type = sys.argv[2]  # "horizontal", "vertical", "slanted non conforming"
        save_folder_root = "./case_1"

        if case_type == "horizontal":
            name_root = "case_" + str(case_id) + "_horizontal"
            output_file_ppu = "./case_1/horizontal_ppu/MASS"
            output_file_hu = "./case_1/horizontal_hu/MASS_OVER_TIME"
            save_folder = save_folder_root + "/horizontal"
            x_ticks = np.array([0, 4, 8, 12, 16, 20])

        if case_type == "vertical":
            name_root = "case_" + str(case_id) + "_vertical"
            output_file_hu = "./case_1/vertical_hu/MASS_OVER_TIME"
            output_file_ppu = "./case_1/vertical_ppu/MASS_OVER_TIME"
            save_folder = save_folder_root + "/vertical"
            x_ticks = np.array([0, 2, 4, 6])

        if case_type == "slanted":
            name_root = "case_" + str(case_id) + "_slanted"
            output_file_ppu = "./case_1/slanted_ppu/MASS_OVER_TIME"
            output_file_hu = "./case_1/slanted_hu/MASS_OVER_TIME"
            save_folder = save_folder_root + "/slanted"
            x_ticks = np.array([0, 2, 4, 6, 8, 10])

        if case_type == "slanted_non_conforming":
            name_root = "case_" + str(case_id) + "_slanted_non_conforming"
            output_file_ppu = "./case_1/slanted_ppu/non-conforming/MASS_OVER_TIME"
            output_file_hu = "./case_1/slanted_hu/non-conforming/MASS_OVER_TIME"
            save_folder = save_folder_root + "/slanted-non-conforming"
            x_ticks = np.array([0, 2, 4, 6, 8, 10])

    if case_id == 2:
        name_root = "case_" + str(case_id)
        save_folder = "./case_2"
        output_file_ppu = "./case_2/ppu/MASS_OVER_TIME"
        output_file_hu = "./case_2/hu/MASS_OVER_TIME"
        x_ticks = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05])

    if case_id in np.array([3, 6, 7, 8]):
        name_root = "case_" + str(case_id)
        save_folder = "./" + name_root  # dont ask me why...
        output_file_ppu = "./" + name_root + "/ppu/MASS_OVER_TIME"
        output_file_hu = "./" + name_root + "/hu/MASS_OVER_TIME"
        x_ticks = np.array([0, 20, 40, 60, 80, 100])

    os.system("mkdir " + save_folder)

    fontsize = 28
    my_orange = "darkorange"
    my_blu = [0.1, 0.1, 0.8]

    time_hu, mass_hu = load_output_mass(output_file_hu)
    delta_mass_hu = np.abs(mass_hu - mass_hu[0])

    # np.set_printoptions(precision=16)
    # print(delta_mass_hu)

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("font", size=fontsize)

    params = {"text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}"}
    plt.rcParams.update(params)
    matplotlib.rcParams["axes.linewidth"] = 1.5

    fig, ax_1 = plt.subplots()

    for label in ax_1.get_xticklabels() + ax_1.get_yticklabels():
        label.set_fontsize(fontsize)

    ax_1.plot(
        time_hu,
        delta_mass_hu,
        label="$|\Delta m|$",
        linestyle="-",
        color=my_blu,
        marker="",
    )

    ax_1.set_xlabel("time", fontsize=fontsize)
    ax_1.set_ylabel("$|\Delta m|$", fontsize=fontsize)
    ax_1.set_xticks(x_ticks)

    ax_1.set_ylim([1.235e-14, 1.242e-14])  ### HARDOCODED FOR SLANTED

    ax_1.grid(linestyle="--", alpha=0.5)

    plt.savefig(
        save_folder + "/" + name_root + "_mass.pdf",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    # we don't need it for this figure...
    # # legend:
    # handles_all, labels_all = [
    #     (a + b)
    #     for a, b in zip(
    #         ax_1.get_legend_handles_labels(), ax_1.get_legend_handles_labels()
    #     )
    # ]

    # handles = np.ravel(np.reshape(handles_all[:2], (1, 2)), order="F")
    # labels = np.ravel(np.reshape(labels_all[:2], (1, 2)), order="F")

    # fig, ax = plt.subplots(figsize=(25, 10))
    # for h, l in zip(handles, labels):
    #     ax.plot(np.zeros(1), label=l)

    # ax.legend(
    #     handles,
    #     labels,
    #     fontsize=fontsize,
    #     loc="lower center",
    #     ncol=1,
    #     bbox_to_anchor=(-0.1, -0.65),
    # )

    # filename = save_folder + "/" + name_root + "_mass" + "_legend.pdf"
    # fig.savefig(filename, bbox_inches="tight")
    # plt.gcf().clear()

    # os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
    # os.system("pdfcrop " + filename + " " + filename)

    print("\nDone!")
