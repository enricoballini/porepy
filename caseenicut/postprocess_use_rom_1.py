import sys
import os
import pdb

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

sys.path.append("/home/inspiron/Desktop/PhD/datapaper4-eni/mypythonmodules")
sys.path.append("/home/inspiron/Desktop/PhD/datapaper4-eni/eni_venv/porepy/src")

import porepy as pp


os.system("clear")

print("# normal_area = np.load(./data/mech/0/normal_area.npy)  ### TODO")


def compute_cff(tractions):
    """ """
    friction_coeff = 0.45
    dim = 3
    # normal_area = np.load("./data/mech/0/normal_area.npy")  ### TODO #######################################################################
    normal_area = np.array([1, 1, 1])
    normal = normal_area / np.linalg.norm(normal_area, ord=2)

    normal_projection = pp.map_geometry.normal_matrix(normal=normal)
    tangential_projetion = pp.map_geometry.tangent_matrix(normal=normal)
    T_vect_frac = np.reshape(tractions, (dim, -1), order="F")

    T_normal = normal_projection @ T_vect_frac
    T_normal_normal = np.dot(normal, T_normal)
    # T_normal_norm = np.linalg.norm(T_normal.T, ord=2, axis=1)
    T_tangential = tangential_projetion @ T_vect_frac
    # T_tangential_y = T_tangential[1].T  # one on-plane direction is aligned with y
    # T_tangential_t = np.linalg.norm(
    #     np.array([T_tangential[0], T_tangential[2]]).T, ord=2, axis=1
    # )  # the other tangential dir, t, is made of x and z
    T_tangential_norm = np.linalg.norm(T_tangential.T, ord=2, axis=1)

    cff = T_tangential_norm - friction_coeff * T_normal_normal
    return cff


def plot_cumulative(
    cff_x,
    cff_cumulative,
    names,
    time,
    save_folder,
):
    """ """
    fontsize = 24
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("font", size=fontsize)

    params = {
        "text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}\usepackage{mathrsfs}"
    }
    plt.rcParams.update(params)
    matplotlib.rcParams["axes.linewidth"] = 1.5

    n_lines = len(cff_cumulative)

    fig, ax_1 = plt.subplots()

    for label in ax_1.get_xticklabels() + ax_1.get_yticklabels():
        label.set_fontsize(fontsize)

    for i in np.arange(n_lines):
        ax_1.plot(
            cff_x,
            cff_cumulative[i],
            label=names[i],
            linestyle="-",
            color=[0.3 + 0.7 * i / n_lines, 0.5, 0.5],
            marker="",
        )

    ax_1.set_xscale("log")
    ax_1.set_xlabel("CFF", fontsize=fontsize)
    # ax_1.set_xticks(x_ticks)

    ax_1.grid(linestyle="-", alpha=0.5)

    plt.savefig(
        save_folder + "/cff_cumulative_" + str(time) + ".pdf",
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

    handles = np.ravel(
        np.reshape(handles_all[: 1 * n_lines], (1, 1 * n_lines)), order="F"
    )
    labels = np.ravel(
        np.reshape(labels_all[: 1 * n_lines], (1, 1 * n_lines)), order="F"
    )
    fig, ax = plt.subplots(figsize=(25, 10))
    for h, l in zip(handles, labels):
        ax.plot(np.zeros(1), label=l)

    ax.legend(
        handles,
        labels,
        fontsize=fontsize,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(-0.1, -0.65),
    )

    filename = save_folder + "/cff_cumulative_" + str(time) + "_label.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.gcf().clear()

    os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
    os.system("pdfcrop " + filename + " " + filename)


def create_evaluation_matrix(n_pts_per_axis):
    """ """
    discretizations = [None] * n_params
    for i in range(n_params):
        discretizations[i] = np.linspace(
            parameters_range_cut[0, i], parameters_range_cut[1, i], n_pts_per_axis
        )
    hyper_cube = np.meshgrid(*discretizations)
    return hyper_cube


times = np.array([7305, 14610])

parameters_range = np.array(
    [
        [np.log(1e-1), np.log(1e-1), 1, 5.71e9, 1.0, 1.0e6, 703000.0],
        [np.log(1e2), np.log(1e1), 25, 5.71e11, 1.0, 1.5e6, 703000.0],
    ]
)  # same as training

parameters_range_cut = parameters_range[:, [0, 1, 2, 3, 5]]
n_params = 5


nn = torch.load("./results/nn/model_trained.pth")
nn.set_forward_mode("online")
nn.to("cuda")
nn.eval()

n_pts_per_axis_list = np.array([3, 4, 5, 6])
print("max number of evaluations: ", n_pts_per_axis_list**n_params)

for time in times:
    print("\n\n\n time = ", time, "\n")

    cff_cumulative = [None] * len(n_pts_per_axis_list)
    for ii, n_pts_per_axis in enumerate(n_pts_per_axis_list):
        hyper_cube = create_evaluation_matrix(n_pts_per_axis)
        param_list = list(zip(*[X.ravel() for X in hyper_cube]))
        max_cffs = np.zeros(len(param_list))

        for i, param in enumerate(param_list):
            tractions = (
                nn(torch.tensor(np.append(param, time), dtype=torch.float32), "cuda")
                .detach()
                .cpu()
                .numpy()
            )
            cff = compute_cff(tractions)
            max_cffs[i] = np.max(cff)

        n_pts_plot = 1000
        x_discr = np.linspace(1e4, 3e6, n_pts_plot)
        y = np.zeros(n_pts_plot)

        for i, x in enumerate(x_discr):
            y[i] = np.sum(max_cffs > x) / len(param_list)

        cff_cumulative[ii] = y

    plot_cumulative(
        x_discr,
        cff_cumulative,
        names=n_pts_per_axis_list.astype("str"),
        time=time,
        save_folder="./results/nn",
    )

print("\n\n# normal_area = np.load(./data/mech/0/normal_area.npy)  ### TODO")
print("\nDone!")
