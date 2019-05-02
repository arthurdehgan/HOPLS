from itertools import product
from math import ceil
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import loadmat
from sklearn import mixture
import numpy as np

MINMAX = [0, 1]
Y_LABEL = "Q2"
COLORS = list(sns.color_palette("deep"))
WIDTH = .90
GRAPH_TITLE = "comparaison of PLS algorithms"

RESOLUTION = 300


def autolabel(ax, rects):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        width = rect.get_width()
        this_color = "black"

        if height <= MINMAX[0]:
            ax.text(
                rect.get_x() + width / 2.,
                width + 1. * height,
                "%d" % int(height),
                ha="center",
                va="bottom",
                color=this_color,
                size=11,
            )
    return ax


def generate_barplot(X_mode, Y_mode, ss, name):
    labels = ["HOPLS", "NPLS", "PLS"]
    groups = ["5dB", "0dB", "-2dB", "-5dB"]

    nb_labels = len(labels)
    dat = []
    stds = []
    for lab in labels:
        file_name = lab + _
        temp = loadmat(file_name)["q2_test"].squeeze()[:4]
        if len(temp.shape) > 1:
            stds.append(np.std(temp[:, :4], axis=1))
            temp = np.mean(temp[:, :4], axis=1)
        else:
            stds.append([0, 0, 0, 0])
        dat.append(temp)

    dat = np.asarray(dat).T
    stds = np.asarray(stds).T
    fig = plt.figure(figsize=(10, 5))  # size of the figure

    # Generating the barplot (do not change)
    ax = plt.axes()
    temp = 0
    for group in range(len(groups)):
        bars = []
        data = dat[group]
        ts = stds[group]
        for i, val in enumerate(data):
            pos = i + 1
            color = COLORS[i]
            bars.append(ax.bar(temp + pos, val, WIDTH, color=color, yerr=ts[i]))
            ax = autolabel(ax, bars[i])
        temp += pos + 1

    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(bottom=MINMAX[0], top=MINMAX[1])
    ax.set_title(GRAPH_TITLE)
    ax.set_xticklabels(groups)
    ax.set_xticks(
        [ceil(nb_labels / 2) + i * (1 + nb_labels) for i in range(len(groups))]
    )
    ax.legend(
        bars,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=False,
        shadow=False,
        ncol=len(labels),
    )

    save_name = "barplot.png"
    fig.savefig(save_name, dpi=RESOLUTION)


if __name__ == "__main__":
    modeX = [3, 5]
    modeY = [2, 3]
    sample_sizes = [10, 20]
    names = ["", "complex_"]
    for X_mode, Y_mode, ss, name in product(modeX, modeY, sample_sizes, names):
        generate_barplot(X_mode.Y_mode, ss, name)
