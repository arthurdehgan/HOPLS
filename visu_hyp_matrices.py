from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
import numpy as np

LOADPATH = "results/"
methods = ["sinc"]


def generate_matrix(X_mode, Y_mode, ss, name, lab="HOPLS"):
    groups = ["5dB", "0dB", "-2dB", "-5dB"]
    plt.figure(figsize=(40, 10))
    for interp_method in methods:
        filename = LOADPATH + lab + f"_results_{name}X{X_mode}_{Y_mode}_ss{ss}"
        dat = np.mean(loadmat(filename)["hyp"], axis=(1, 2))
        fig, axs = plt.subplots(
            nrows=1, ncols=4, figsize=(9.3, 2), subplot_kw={"xticks": [], "yticks": []}
        )
        for ax, stuff in zip(axs, enumerate(dat)):
            i, grid = stuff
            grid[grid < 0] = 0
            test = np.mean(grid)
            grid[grid < test] = test

            fig.subplots_adjust(left=0.03, right=0.5, hspace=0.3, wspace=0.05)

            im = ax.imshow(grid[:, 4:15], interpolation=interp_method, cmap="viridis")
            ax.set_title(str(groups[i]))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(im, cax=cax)
            ax.set_yticks(np.arange(0, 8, 2))
            ax.set_xticks(np.arange(0, 11, 2))
            ax.set_xticklabels(np.arange(4, 15, 2))
            ax.set_yticklabels(np.arange(2, 10, 2))
            ax.set_xlabel("R")
            if ax == axs[0]:
                ax.set_ylabel(r"$\lambda$")

        plt.tight_layout()
        filename = f"4{name}s{ss}_X{X_mode}_Y{Y_mode}_matrices_{interp_method}.png"
        plt.savefig(filename, dpi=300)


if __name__ == "__main__":
    modeX = [3, 5]
    modeY = [2, 3]
    sample_sizes = [20]
    name = ""
    for X_mode, Y_mode, ss in product(modeX, modeY, sample_sizes):
        generate_matrix(X_mode, Y_mode, ss, name)
