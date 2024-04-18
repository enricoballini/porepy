import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import pdb


os.system("clear")


def load_s_at_timestep(timestep, method, data_folder):
    """
    - saturation is the 10 column
    - y is the last
    """
    saturation = []
    y = []
    with open(
        data_folder + "/case_1_hor_diff_" + method + "_" + str(timestep) + ".txt", "r"
    ) as f:
        lines = f.read().splitlines()[1:]
        lines = np.array([lines[i].split(",") for i in range(len(lines))])
        saturation = lines[:, 9].astype(np.float32)
        y = lines[:, -1].astype(np.float32)

    return saturation, y


def make_figure(s_list, y_list, timestep, labels):
    """ """
    colors = [
        "darkorange",
        np.array([0.1, 0.1, 0.8]),
    ]

    fontsize = 28
    x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    y_ticks = [0, 0.3, 0.5, 1]

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("font", size=fontsize)

    params = {"text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}"}
    plt.rcParams.update(params)
    matplotlib.rcParams["axes.linewidth"] = 1.5

    fig, ax_1 = plt.subplots()

    for label in ax_1.get_xticklabels() + ax_1.get_yticklabels():
        label.set_fontsize(fontsize)

    i = 0
    for s, y, label, color in zip(s_list, y_list, labels, colors):
        ax_1.plot(
            s,
            y,
            label=label,
            linestyle="-",
            color=color,
            marker="",
        )

    ax_1.set_xlabel("$S_0$", fontsize=fontsize)
    ax_1.set_ylabel("$y$", fontsize=fontsize)
    ax_1.set_xticks(x_ticks)
    ax_1.set_yticks(y_ticks)

    ax_1.grid(linestyle="--", alpha=0.5)

    plt.savefig(
        save_folder + "/case_1_hor_saturation_" + str(timestep) + ".pdf",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    # legend:
    handles_all, labels_all = [
        (a + b)
        for a, b in zip(
            ax_1.get_legend_handles_labels(), ax_1.get_legend_handles_labels()
        )
    ]

    handles = np.ravel(np.reshape(handles_all[:2], (1, 2)), order="F")
    labels = np.ravel(np.reshape(labels_all[:2], (1, 2)), order="F")

    fig, ax = plt.subplots(figsize=(25, 10))
    for h, l in zip(handles, labels):
        ax.plot(np.zeros(1), label=l)

    ax.legend(
        handles,
        labels,
        fontsize=fontsize,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(-0.1, -0.65),
    )

    filename = (
        save_folder + "/" + "/case_1_hor_saturation_legend_" + str(timestep) + ".pdf"
    )
    fig.savefig(filename, bbox_inches="tight")
    plt.gcf().clear()

    os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
    os.system("pdfcrop " + filename + " " + filename)


data_folder = "./figures"
save_folder = "./figures"

s_17_ppu, y_17_ppu = load_s_at_timestep(17, "ppu", data_folder)
s_17_hu, y_17_hu = load_s_at_timestep(17, "hu", data_folder)
y_list = [y_17_ppu, y_17_hu]
s_list = [s_17_ppu, s_17_hu]
labels = ["$PPU$", "$HU$"]
make_figure(s_list, y_list, 17, labels)


s_51_ppu, y_51_ppu = load_s_at_timestep(51, "ppu", data_folder)
s_51_hu, y_51_hu = load_s_at_timestep(51, "hu", data_folder)
y_list = [y_51_ppu, y_51_hu]
s_list = [s_51_ppu, s_51_hu]
labels = ["$PPU$", "$HU$"]
make_figure(s_list, y_list, 51, labels)

# 17 -> time = 6.8
# 51 -> time = 20

print("\nDone!")
