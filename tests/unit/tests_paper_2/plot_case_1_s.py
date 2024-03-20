import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import pdb


os.system("clear")


def load_s_at_timestep(timestep, data_folder):
    """
    - saturation is the 6 column
    - y is the last
    """
    saturation = []
    y = []
    with open(data_folder + "/" + str(timestep) + ".txt", "r") as f:
        lines = f.read().splitlines()[1:]
        lines = np.array([lines[i].split(",") for i in range(len(lines))])
        saturation = lines[:, 5].astype(np.float32)
        y = lines[:, -1].astype(np.float32)

    return saturation, y


data_folder = "./figures"
save_folder = "./figures"

s_0, y_0 = load_s_at_timestep(0, data_folder)
s_3, y_3 = load_s_at_timestep(3, data_folder)
s_20, y_20 = load_s_at_timestep(20, data_folder)

s_list = [s_0, s_3, s_20]
y_list = [y_0, y_3, y_20]
labels = ["$t = 0$", "$t = 0.3$", "$t = 2$"]
colors = [
    np.array([0.2, 0.2, 0.2]),
    np.array([0.7, 0.6, 0.5]),
    np.array([0.9, 0.8, 0.5]),
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
    save_folder + "/case_1_vertical_saturation.pdf",
    dpi=150,
    bbox_inches="tight",
    pad_inches=0.2,
)

# legend:
handles_all, labels_all = [
    (a + b)
    for a, b in zip(ax_1.get_legend_handles_labels(), ax_1.get_legend_handles_labels())
]

handles = np.ravel(np.reshape(handles_all[:3], (1, 3)), order="F")
labels = np.ravel(np.reshape(labels_all[:3], (1, 3)), order="F")

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

filename = save_folder + "/" + "/case_1_vertical_saturation_legend.pdf"
fig.savefig(filename, bbox_inches="tight")
plt.gcf().clear()

os.system("pdfcrop --margins '0 -800 0 0' " + filename + " " + filename)
os.system("pdfcrop " + filename + " " + filename)

print("\nDone!")
