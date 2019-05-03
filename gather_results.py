from itertools import product
import numpy as np
from scipy.io import loadmat, savemat

noise = [5, 0, -2, -5]
modeX = [3, 5]
modeY = [2, 3]
sample_size = [10, 20]
names = [""]
n_datasets = 24

for X_mode, Y_mode, name, ss in product(modeX, modeY, names, sample_size):
    print(name, X_mode, Y_mode, ss)
    fPLS_scores, fNPLS_scores, fHOPLS_scores = [], [], []
    fPLS_R, fNPLS_R, fHOPLS_R = [], [], []
    fHOPLS_L = []
    fNPLS_hyp, fHOPLS_hyp, fPLS_hyp = [], [], []
    for snr in noise:
        PLS_scores, NPLS_scores, HOPLS_scores = [], [], []
        PLS_R, NPLS_R, HOPLS_R = [], [], []
        HOPLS_L = []
        NPLS_hyp, HOPLS_hyp, PLS_hyp = [], [], []
        for n in range(n_datasets):
            filename = f"./results/res{n}_s{ss}_X{X_mode}_Y{Y_mode}_{snr}dB.mat"
            dat = loadmat(filename)
            PLS_hyp += [dat["PLS_hyp"]]
            NPLS_hyp += [dat["NPLS_hyp"]]
            HOPLS_hyp += [dat["HOPLS_hyp"]]
            PLS_scores += dat["PLS_Q2"].flatten().tolist()
            NPLS_scores += dat["NPLS_Q2"].flatten().tolist()
            HOPLS_scores += dat["HOPLS_Q2"].flatten().tolist()
            PLS_R += dat["PLS_R"].flatten().tolist()
            NPLS_R += dat["NPLS_R"].flatten().tolist()
            HOPLS_R += dat["HOPLS_R"].flatten().tolist()
            HOPLS_L += dat["HOPLS_L"].flatten().tolist()
        fPLS_hyp.append(PLS_hyp)
        fNPLS_hyp.append(NPLS_hyp)
        fHOPLS_hyp.append(HOPLS_hyp)
        fPLS_scores.append(PLS_scores)
        fNPLS_scores.append(NPLS_scores)
        fHOPLS_scores.append(HOPLS_scores)
        fPLS_R.append(PLS_R)
        fNPLS_R.append(NPLS_R)
        fHOPLS_R.append(HOPLS_R)
        fHOPLS_L.append(HOPLS_L)
    savemat(
        f"results/PLS_results_{name}X{X_mode}_{Y_mode}_ss{ss}",
        {
            "R": np.asarray(fPLS_R),
            "Q2": np.asarray(fPLS_scores),
            "hyp": np.asarray(fPLS_hyp),
        },
    )
    savemat(
        f"results/NPLS_results_{name}X{X_mode}_{Y_mode}_ss{ss}",
        {
            "R": np.asarray(fNPLS_R),
            "Q2": np.asarray(fNPLS_scores),
            "hyp": np.asarray(fNPLS_hyp),
        },
    )
    savemat(
        f"results/HOPLS_results_{name}X{X_mode}_{Y_mode}_ss{ss}",
        {
            "R": np.asarray(fHOPLS_R),
            "L": np.asarray(fHOPLS_L),
            "Q2": np.asarray(fHOPLS_scores),
            "hyp": np.asarray(fHOPLS_hyp),
        },
    )
