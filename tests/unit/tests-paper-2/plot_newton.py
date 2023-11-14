import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import pdb


os.system("clear")


def load_output_newton(filename):
    """ """
    data = np.loadtxt(filename, delimiter=",")
    return (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        data[:, 4],
        data[:, 5],
        data[:, 6],
    )


#####################################################

# output_file_hu = "./visualization_test_0_hu/OUTPUT_NEWTON_INFO"
# output_file_ppu = "./visualization_test_0_ppu/OUTPUT_NEWTON_INFO"

output_file_hu = "./visualization-test-1-hu/OUTPUT_NEWTON_INFO"
output_file_ppu = "./visualization-test-1-ppu/OUTPUT_NEWTON_INFO"

fontsize = 28
my_orange = "darkorange"
my_blu = [0.1, 0.1, 0.8]

x_ticks = np.array([0, 10, 20, 30, 40, 50])

save_folder = "./"


#####################################################
(
    time_ppu,
    time_steps_ppu,
    time_chops_ppu,
    cumulative_iterations_ppu,
    global_cumulative_iterations_ppu,
    last_iterations_ppu,
    wasted_iterations_ppu,
) = load_output_newton(output_file_ppu)

(
    time_hu,
    time_steps_hu,
    time_chops_hu,
    cumulative_iterations_hu,
    global_cumulative_iterations_hu,
    last_iterations_hu,
    wasted_iterations_hu,
) = load_output_newton(output_file_hu)


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
    time_ppu,
    global_cumulative_iterations_ppu,
    label="$K_{ppu}$",
    linestyle="--",
    color=my_orange,
    marker="",
)
ax_1.plot(
    time_hu,
    global_cumulative_iterations_hu,
    label="$K_{hu}$",
    linestyle="-",
    color=my_blu,
    marker="",
)

ax_1.set_xlabel("time $[s]$", fontsize=fontsize)
ax_1.set_xticks(x_ticks)

ax_1.grid(linestyle="--", alpha=0.5)

plt.savefig(
    save_folder + "/global_cumulative.pdf",
    dpi=150,
    bbox_inches="tight",
    pad_inches=0.2,
)


# legend:
handles_all, labels_all = [
    (a + b)
    for a, b in zip(ax_1.get_legend_handles_labels(), ax_1.get_legend_handles_labels())
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
    ncol=3,
    bbox_to_anchor=(-0.1, -0.65),
)

filename = save_folder + "/global_cumulative" + "_legend.pdf"
fig.savefig(filename, bbox_inches="tight")
plt.gcf().clear()

os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
os.system("pdfcrop " + filename + " " + filename)


pdb.set_trace()


print("\n\n\nDone!")
