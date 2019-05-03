from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
import numpy as np

LOADPATH = "results/"


def generate_line(X_mode, Y_mode, ss, name, lab="HOPLS"):
    groups = ["5dB", "0dB", "-2dB", "-5dB"]
    filename = LOADPATH + lab + f"_results_{name}X{X_mode}_{Y_mode}_ss{ss}"
    dat = np.mean(loadmat(filename)["hyp"], axis=(1, 2))
    fig, axs = plt.subplots(
        nrows=2, ncols=4, figsize=(8, 3), subplot_kw={"xticks": [], "yticks": []}
    )
    i = 0
    for j in range(4):
        for i in range(2):
            grid = dat[j][:, :20]
            grid[grid < 0] = 0
            line = np.mean(grid, axis=i)

            fig.subplots_adjust(left=0.03, right=0.8, hspace=0.3, wspace=0.05)

            axs[i, j].plot(line)
            if i == 0:
                gen = np.arange(np.min(line), np.max(line) + .2, .1)
                used = (
                    gen
                    if len(gen) < 6
                    else np.arange(np.min(line), np.max(line) + .2, .2)
                )
                axs[i, j].set_yticks(used)
                axs[i, j].set_yticklabels([f"{k:.1f}" for k in used])
                axs[i, j].set_xticks([0, 4, 9, 14, 19])
                axs[i, j].set_xticklabels([1, 5, 10, 15, 20])
                axs[i, j].set_title(str(groups[j]))
                axs[i, j].set_xlabel("R")
                if j == 0:
                    axs[i, j].set_ylabel(r"$Q^2$")
            if i == 1:
                gen = np.arange(np.min(line), np.max(line) + .2, .1)
                used = (
                    gen
                    if len(gen) < 6
                    else np.arange(np.min(line), np.max(line) + .2, .2)
                )
                axs[i, j].set_yticks(used)
                axs[i, j].set_yticklabels([f"{k:.1f}" for k in used])
                axs[i, j].set_xticks(np.arange(0, 8, 1))
                axs[i, j].set_xticklabels(np.arange(2, 10, 1))
                axs[i, j].set_xlabel(r"$\lambda$")
                if j == 0:
                    axs[i, j].set_ylabel(r"$Q^2$")

    plt.tight_layout()
    filename = f"{name}s{ss}_X{X_mode}_Y{Y_mode}_lines.png"
    plt.savefig(filename)


if __name__ == "__main__":
    modeX = [3, 5]
    modeY = [2, 3]
    sample_sizes = [10, 20]
    name = ""
    for X_mode, Y_mode, ss in product(modeX, modeY, sample_sizes):
        generate_line(X_mode, Y_mode, ss, name)
