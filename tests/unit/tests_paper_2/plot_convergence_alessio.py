import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

import os

fontsize = 24
my_orange = "darkorange"
my_blu = [0.1, 0.1, 0.8]

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=fontsize)

params = {"text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}"}
plt.rcParams.update(params)
matplotlib.rcParams["axes.linewidth"] = 1.5


######################################################à


def load_data(root_paths, mesh_type, kind):
    err_list_p_2d = [np.array([])] * len(root_paths)
    err_list_p_1d = [np.array([])] * len(root_paths)
    err_list_s_2d = [np.array([])] * len(root_paths)
    err_list_s_1d = [np.array([])] * len(root_paths)
    err_list_mortar_0 = [np.array([])] * len(root_paths)
    err_list_mortar_1 = [np.array([])] * len(root_paths)

    for index, root_path in enumerate(root_paths):
        cell_sizes = np.loadtxt(root_path + "cell_sizes")  # last one is the ref

        # fine (ref): ------------------------------------------------------
        cell_size = cell_sizes[-1]
        variable_num_dofs = np.loadtxt(
            root_path + "variable_num_dofs_" + str(cell_size)
        )

        volumes_2d_ref = np.load(
            root_path + "volumes_2d_" + str(cell_size) + ".npy", allow_pickle=True
        )
        volumes_1d_ref = np.loadtxt(root_path + "volumes_1d_" + str(cell_size))
        volumes_mortar = np.concatenate(
            (volumes_1d_ref, volumes_1d_ref)
        )  # this holds also for non-matching subdomains!

        id_2d = np.arange(variable_num_dofs[0], dtype=np.int32)  # HARDCODED
        id_1d = np.arange(
            variable_num_dofs[0],
            variable_num_dofs[0] + variable_num_dofs[1],
            dtype=np.int32,
        )  # HARDCODED

        pressure = np.load(
            root_path + "pressure_" + str(cell_size) + ".npy", allow_pickle=True
        )
        saturation = np.load(
            root_path + "saturation_" + str(cell_size) + ".npy", allow_pickle=True
        )
        mortar_phase_0_ref = np.loadtxt(root_path + "mortar_phase_0_" + str(cell_size))
        mortar_phase_1_ref = np.loadtxt(root_path + "mortar_phase_1_" + str(cell_size))

        pressure_2d_ref = pressure[id_2d]
        saturation_2d_ref = saturation[id_2d]
        pressure_1d_ref = pressure[id_1d]
        saturation_1d_ref = saturation[id_1d]

        for cell_size in cell_sizes[:-1]:
            # coarse: --------------------------------------------------------
            variable_num_dofs = np.loadtxt(
                root_path + "variable_num_dofs_" + str(cell_size)
            )

            coarse_to_fine_2d = sp.sparse.load_npz(
                root_path + "mapping_matrix_2d_" + str(cell_size) + ".npz"
            )
            coarse_to_fine_1d = sp.sparse.load_npz(
                root_path + "mapping_matrix_1d_" + str(cell_size) + ".npz"
            )  #

            coarse_to_fine_intf = sp.sparse.kron(
                sp.sparse.eye(2), coarse_to_fine_1d
            )  # sorry, I want to use kron...
            # intf is conforming with 1d

            pressure = np.load(root_path + "pressure_" + str(cell_size) + ".npy")
            saturation = np.load(root_path + "saturation_" + str(cell_size) + ".npy")
            mortar_phase_0 = np.loadtxt(root_path + "mortar_phase_0_" + str(cell_size))
            mortar_phase_1 = np.loadtxt(root_path + "mortar_phase_1_" + str(cell_size))

            id_2d = np.arange(variable_num_dofs[0], dtype=np.int32)  # HARDCODED
            id_1d = np.arange(
                variable_num_dofs[0],
                variable_num_dofs[0] + variable_num_dofs[1],
                dtype=np.int32,
            )  # HARDCODED

            pressure_2d_proj = coarse_to_fine_2d @ pressure[id_2d]
            pressure_1d_proj = coarse_to_fine_1d @ pressure[id_1d]
            saturation_2d_proj = coarse_to_fine_2d @ saturation[id_2d]
            saturation_1d_proj = coarse_to_fine_1d @ saturation[id_1d]

            mortar_phase_0_proj = coarse_to_fine_intf @ mortar_phase_0
            mortar_phase_1_proj = coarse_to_fine_intf @ mortar_phase_1

            err_p_2d = np.linalg.norm(
                (pressure_2d_proj - pressure_2d_ref) * volumes_2d_ref, ord=2
            ) / np.linalg.norm(pressure_2d_ref * volumes_2d_ref, ord=2)

            err_p_1d = np.linalg.norm(
                (pressure_1d_proj - pressure_1d_ref) * volumes_1d_ref, ord=2
            ) / np.linalg.norm(pressure_1d_ref * volumes_1d_ref, ord=2)

            err_s_2d = np.linalg.norm(
                (saturation_2d_proj - saturation_2d_ref) * volumes_2d_ref, ord=2
            ) / np.linalg.norm(saturation_2d_ref * volumes_2d_ref, ord=2)

            err_s_1d = np.linalg.norm(
                (saturation_1d_proj - saturation_1d_ref) * volumes_1d_ref, ord=2
            ) / np.linalg.norm(saturation_1d_ref * volumes_1d_ref, ord=2)

            norm_perm = 0.01
            err_mortar_0 = np.linalg.norm(
                (mortar_phase_0_proj - mortar_phase_0_ref) * volumes_mortar, ord=2
            ) / np.linalg.norm(norm_perm * volumes_mortar, ord=2)

            err_mortar_1 = np.linalg.norm(
                (mortar_phase_1_proj - mortar_phase_1_ref) * volumes_mortar, ord=2
            ) / np.linalg.norm(norm_perm * volumes_mortar, ord=2)

            err_list_p_2d[index] = np.append(err_list_p_2d[index], err_p_2d)
            err_list_p_1d[index] = np.append(err_list_p_1d[index], err_p_1d)
            err_list_s_2d[index] = np.append(err_list_s_2d[index], err_s_2d)
            err_list_s_1d[index] = np.append(err_list_s_1d[index], err_s_1d)
            err_list_mortar_0[index] = np.append(err_list_mortar_0[index], err_mortar_0)
            err_list_mortar_1[index] = np.append(err_list_mortar_1[index], err_mortar_1)

    if kind == "pressure":
        items = {
            mesh_type: zip(
                [
                    ["err_list_p_2d", "$p_{h}$"],
                    ["err_list_p_1d", "$p_{l}$"],
                ],
                [
                    err_list_p_2d,
                    err_list_p_1d,
                ],
            )
        }
    elif kind == "saturation":
        items = {
            mesh_type: zip(
                [
                    ["err_list_s_2d", "$S_{0,h}$"],
                    ["err_list_s_1d", "$S_{0,l}$"],
                ],
                [
                    err_list_s_2d,
                    err_list_s_1d,
                ],
            )
        }
    elif kind == "mortar":
        items = {
            mesh_type: zip(
                [
                    ["err_list_mortar_0", "$\zeta_0$"],
                    ["err_list_mortar_1", "$\zeta_1$"],
                ],
                [
                    err_list_mortar_0,
                    err_list_mortar_1,
                ],
            )
        }

    return items, cell_sizes


#####################################################


def main_plot(
    save_folder,
    all_items,
    cell_sizes,
    coeff,
):

    color = {"conforming": my_orange, "non_conforming": my_blu}
    my_label = {"conforming": "conf ", "non_conforming": "non-conf "}
    linestyle = {
        "2d": "--",
        "1d": "-",
        "_0": "--",
        "_1": "-",
    }  # per i mortar 0 e 1 riguardano la fase 0 e 1

    xticks = (0.022, 0.065, 0.175)
    xticklabels = (
        "$2.2\\times 10^{-2}$",
        "$6.5\\times 10^{-2}$",
        "$1.75\\times 10^{-1}$",
    )

    fig, ax_1 = plt.subplots()
    for label in ax_1.get_xticklabels() + ax_1.get_yticklabels():
        label.set_fontsize(20)

    for mesh_type, items in all_items.items():
        for names, val in items:
            name = names[0]
            name_label = names[1]

            ax_1.loglog(
                cell_sizes[:-1],
                val[0],
                label=my_label[mesh_type] + name_label,
                linestyle=linestyle[name[-2:]],
                color=color[mesh_type],
                marker="o",
            )
            ax_1.loglog(
                cell_sizes[:-1],
                coeff * cell_sizes[:-1],
                label="$\\mathcal{O}(h)$",
                linestyle="--",
                color=[0, 0, 0],
                marker="",
            )

            x = np.mean(0.6 * cell_sizes[:-1])
            y = np.mean(0.9 * coeff * cell_sizes[:-1])
            ax_1.text(x, y, "$\\mathcal{O}(h)$", size=20)

    ax_1.set_ylabel("$L^2$-error")
    ax_1.set_xlabel("$h$", fontsize=fontsize)
    ax_1.set_xticks(xticks)
    ax_1.set_xticklabels(xticklabels)

    ax_1.grid(color="gray", linestyle="--", linewidth=0.5)  # , which="both")

    # handle = ax_1.get_legend_handles_labels()[0][0]
    # plt.legend([handle], [name_label])

    plt.savefig(
        save_folder + "/convergence_" + name + ".pdf",
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

    labels, mask = np.unique(labels_all, return_index=True)
    handles = np.array(handles_all)[mask]

    fig, ax = plt.subplots(figsize=(25, 10))

    for h, l in zip(handles[1:], labels[1:]):
        ax.plot(np.zeros(1), label=l)

    ax.legend(
        handles[1:],
        labels[1:],
        fontsize=fontsize,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(-0.1, -0.65),
    )

    filename = save_folder + "/convergence_" + name + "_legend.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.gcf().clear()

    os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
    os.system("pdfcrop " + filename + " " + filename)


if __name__ == "__main__":

    main_folder = "./case_1/slanted_hu/"
    root_paths_conf = main_folder + "convergence_results/"
    root_paths_nonconf = main_folder + "non-conforming/convergence_results/"
    coeff = [0.15, 0.5, 0.75]
    kind = ["pressure", "saturation", "mortar"]

    for k, c in zip(kind, coeff):
        items_conf, cell_sizes = load_data([root_paths_conf], "conforming", k)
        items_nonconf, cell_sizes = load_data([root_paths_nonconf], "non_conforming", k)

        items_conf.update(items_nonconf)
        main_plot(main_folder, items_conf, cell_sizes, c)
