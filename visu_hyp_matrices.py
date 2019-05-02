import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
import numpy as np

methods = ["nearest", "sinc"]

groups = ["10dB", "5dB", "0dB", "-5dB"]
plt.figure(figsize=(40, 10))
for interp_method in methods:
    dat = np.mean(loadmat("ho_HOPLS_res.mat")["q2_results"][:4], axis=0)
    fig, axs = plt.subplots(
        nrows=1, ncols=4, figsize=(9.3, 2), subplot_kw={"xticks": [], "yticks": []}
    )
    for ax, stuff in zip(axs, enumerate(dat[:4])):
        i, grid = stuff
        grid[grid < 0] = 0
        test = np.mean(grid)
        grid[grid < test] = test

        fig.subplots_adjust(left=0.03, right=0.5, hspace=0.3, wspace=0.05)

        im = ax.imshow(grid[:, 10:20], interpolation=interp_method, cmap="viridis")
        ax.set_title(str(groups[i]))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        ax.set_yticks(np.arange(0, 9, 1))
        ax.set_xticks(np.arange(0, 10, 1))
        ax.set_xticklabels(np.arange(10, 20, 1))
        ax.set_yticklabels(np.arange(2, 11, 1))
        ax.set_xlabel("R")
        if ax == axs[0]:
            ax.set_ylabel(r"$\lambda$")

    plt.tight_layout()
    plt.savefig(f"matrices_{interp_method}.png")
