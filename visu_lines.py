import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
import numpy as np

methods = ["nearest", "sinc"]

groups = ["10dB", "5dB", "0dB", "-5dB"]
dat = np.mean(loadmat("ho_HOPLS_res.mat")["q2_results"][:4], 0)
fig, axs = plt.subplots(
    nrows=2, ncols=4, figsize=(8, 3), subplot_kw={"xticks": [], "yticks": []}
)
# plt.figure(figsize=(20, 5))
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
                gen if len(gen) < 6 else np.arange(np.min(line), np.max(line) + .2, .2)
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
                gen if len(gen) < 6 else np.arange(np.min(line), np.max(line) + .2, .2)
            )
            axs[i, j].set_yticks(used)
            axs[i, j].set_yticklabels([f"{k:.1f}" for k in used])
            axs[i, j].set_xticks(np.arange(0, 9, 1))
            axs[i, j].set_xticklabels(np.arange(2, 11, 1))
            axs[i, j].set_xlabel(r"$\lambda$")
            if j == 0:
                axs[i, j].set_ylabel(r"$Q^2$")

plt.tight_layout()
plt.savefig(f"lines")
