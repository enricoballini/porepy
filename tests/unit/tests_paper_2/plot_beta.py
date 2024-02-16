import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


import os
import sys
import pdb


os.system("clear")

case_id = int(sys.argv[1])  # 1, 2, 3

if case_id == 1:
    case_type = sys.argv[2]  # "horizontal", "vertical", "slanted non conforming"
    x_ticks = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    y_ticks = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    if case_type == "horizontal":
        times = np.around(
            np.loadtxt("./case_1/horizontal_hu/BETA/BETA_TIME"), decimals=6
        )
        root_name = "./case_1/horizontal_hu/BETA"

    if case_type == "vertical":
        times = np.around(np.loadtxt("./case_1/vertical_hu/BETA/BETA_TIME"), decimals=6)
        root_name = "./case_1/vertical_hu/BETA"

    if case_type == "slanted":
        times = np.around(np.loadtxt("./case_1/slanted_hu/BETA/BETA_TIME"), decimals=6)
        root_name = "./case_1/slanted_hu/BETA"

    if case_type == "slanted_non_conforming":
        times = np.around(
            np.loadtxt("./case_1/slanted_hu/non-conforming/BETA/BETA_TIME"),
            decimals=6,
        )
        root_name = "./case_1/slanted_hu/non-conforming/BETA"

    # who the hell are you?
    # times = np.around(
    #     np.loadtxt("./case_1/horizontal_hu_beta/BETA/BETA_TIME"), decimals=6
    # )
    # root_name = "./case_1/horizontal_hu_beta/BETA"

if case_id == 2:
    times = np.around(np.loadtxt("./case_2/hu/BETA/BETA_TIME"), decimals=6)
    root_name = "./case_2/hu/BETA"
    x_ticks = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    y_ticks = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

if case_id == 3:
    times = np.around(np.loadtxt("./case_3/hu/BETA/BETA_TIME"), decimals=6)
    root_name = "./case_3/hu/BETA"


for time in times:
    print("\ntime = ", time, " ----------------------")

    save_folder = root_name
    output_file = root_name + "/BETA_" + str(time)

    info = np.loadtxt(output_file)

    time = info[0][0]
    x = info[:, 1]
    y = info[:, 2]
    beta = info[:, 3]
    beta_scaled = np.abs(beta - 0.5)  # 0 => no upwind, 0.5 => upwind

    delta_potential = info[:, 4]

    x_plot = np.linspace(min(x), max(x), 200)
    y_plot = np.linspace(min(y), max(y), 200)
    X, Y = np.meshgrid(x_plot, x_plot)

    # beta: --------------------
    interp_1 = LinearNDInterpolator(
        list(zip(x, y)), beta_scaled
    )  # this is wrong, HOW??? it inverts the y coordinates, and maybe more... ???
    Z_1 = interp_1(X, Y)
    plt.pcolormesh(X, Y, Z_1, shading="auto", vmin=0.0, vmax=1.0)
    plt.plot(x, y, "ok", label="input point", markersize=1)
    # plt.legend()
    plt.colorbar()
    plt.axis("equal")
    # plt.show()

    plt.savefig(
        save_folder + "/interp_beta_" + str(time) + ".pdf",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, beta_scaled, marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("beta")
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    plt.savefig(
        save_folder + "/beta_" + str(time) + ".pdf",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    plt.close()

    # # delta_potential: --------------
    # interp_2 = LinearNDInterpolator(list(zip(x, y)), delta_potential)
    # Z_2 = interp_2(X, Y)
    # plt.pcolormesh(X, Y, Z_2, shading="auto", vmin=0.0, vmax=1.0)
    # plt.plot(x, y, "ok", label="input point", markersize=1)
    # # plt.legend()
    # plt.colorbar()
    # plt.axis("equal")
    # # plt.show()

    # plt.savefig(
    #     save_folder + "/delta_potential.pdf",
    #     dpi=150,
    #     bbox_inches="tight",
    #     pad_inches=0.2,
    # )

    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(x, y, beta, marker="o")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("delta_potential")
    # plt.show()


print("\nDone!")
