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


def array_to_scientific_latex(arr):
    latex_list = []
    for num in arr:
        exponent = int(np.floor(np.log10(np.abs(num)))) if num != 0 else 0
        mantissa = num / (10**exponent)
        latex_str = f"${mantissa:.2f}\\times 10^{{{exponent}}}$"
        latex_list.append(latex_str)
    return latex_list


def compute_cff(tractions):
    """ """
    friction_coeff = 0.45
    dim = 3
    normal_area = np.load("./data/mech/0/normal_area.npy")
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

    linestyles = ["--", "dashdot", "-"] * n_lines
    for i in np.arange(n_lines):

        # linestyle = "--"*(i % 2 == 0) + "-"*(i % 2 != 0)
        ax_1.plot(
            cff_x,
            cff_cumulative[i],
            label=names[i],
            linestyle=linestyles[i],
            color=[1 * i / (n_lines - 1), 0.8 * i / (n_lines - 1), 0],
            marker="",
        )

    ax_1.set_xscale("log")
    ax_1.set_xlabel(r"$\Delta$CFF", fontsize=fontsize)
    ax_1.set_ylabel("1-CDF", fontsize=fontsize)
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

    handles = np.ravel(np.reshape(handles_all[:n_lines], (1, n_lines)), order="F")
    labels = np.ravel(np.reshape(labels_all[:n_lines], (1, n_lines)), order="F")
    fig, ax = plt.subplots(figsize=(25, 10))
    for h, l in zip(handles, labels):
        ax.plot(np.zeros(1), label=l)

    ax.legend(
        handles,
        labels,
        fontsize=fontsize,
        loc="lower center",
        ncol=np.ceil(n_lines / 2).astype(int),
        bbox_to_anchor=(-0.1, -0.65),
    )

    filename = save_folder + "/cff_cumulative_" + str(time) + "_label.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.gcf().clear()

    os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
    os.system("pdfcrop " + filename + " " + filename)


def create_evaluation_matrix(n_pts_per_axis, parameters_range):
    """ """
    discretizations = [None] * n_params
    for i in range(n_params):
        discretizations[i] = np.linspace(
            parameters_range[0, i], parameters_range[1, i], n_pts_per_axis
        )
    hyper_cube = np.meshgrid(*discretizations)
    return hyper_cube


times = np.array([7305, 14610])


#  0                 1              2                           3               4       5               6
# Ka multiplier, K_\perp mult, Yung modulus ratio, Young's modulus reservoir, Cm, injection_rate, production_rate
parameters_range = np.array(
    [
        [np.log(1e-1), np.log(1e-1), 1, 5.71e9, 1.0, None, 703000.0],
        [np.log(1e2), np.log(1e1), 25, 5.71e11, 1.0, None, 703000.0],
    ]
)  # same as training

parameters_range_cut = parameters_range[:, [0, 1, 2, 3, 5]]
n_params = 5


nn = torch.load("./results/nn/model_trained.pth")
nn.set_forward_mode("online")
nn.to("cuda")
nn.eval()

injection_rate_list = np.linspace(1.0e6, 1.5e6, 8)

# n_pts_per_axis_list = np.array([3, 4, 5, 6])
n_pts_per_axis_list = np.array([6])
print("max number of evaluations: ", n_pts_per_axis_list**n_params)


cff_cumulative_inj = [None] * len(injection_rate_list)

for time in times:
    print("\n\n\n time = ", time, "\n")

    for jj, injection_rate in enumerate(injection_rate_list):

        cff_cumulative = [None] * len(n_pts_per_axis_list)
        for ii, n_pts_per_axis in enumerate(n_pts_per_axis_list):
            parameters_range_cut[:, -1] = injection_rate
            hyper_cube = create_evaluation_matrix(n_pts_per_axis, parameters_range_cut)
            param_list = list(zip(*[X.ravel() for X in hyper_cube]))
            max_cffs = np.zeros(len(param_list))

            for i, param in enumerate(param_list):
                tractions = (
                    nn(
                        torch.tensor(np.append(param, time), dtype=torch.float32),
                        "cuda",
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                cff = compute_cff(tractions)
                print("\nLOAD IT, IT'S SAVE...")
                max_cffs[i] = np.max(cff)

            n_pts_plot = 1000
            x_discr = np.linspace(1e4, 3e6, n_pts_plot)
            y = np.zeros(n_pts_plot)

            for i, x in enumerate(x_discr):
                y[i] = np.sum(max_cffs > x) / len(param_list)

            cff_cumulative[ii] = y

        cff_cumulative_inj[jj] = cff_cumulative[
            0
        ]  # non voglio troppe linee in questo grafico, quindi non valuto la convergenza sul numero di campioni per asse

    #
    plot_cumulative(
        x_discr,
        cff_cumulative_inj,
        names=array_to_scientific_latex(injection_rate_list),
        time=time,
        save_folder="./results/nn",
    )

print("\nDone!")
