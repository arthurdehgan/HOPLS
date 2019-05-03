import os
import sys
import warnings
import torch
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from scipy.io import loadmat, savemat
from joblib import Parallel, delayed
from hopls import matricize, qsquared, HOPLS


def compute_q2_pls(tdata, tlabel, vdata, vlabel, Rval):
    test = PLSRegression(n_components=Rval)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test.fit(matricize(tdata), matricize(tlabel))
    Y_pred = test.predict(matricize(vdata))
    Q2 = qsquared(matricize(vlabel), matricize(Y_pred))
    return Q2


def compute_q2_hopls(tdata, tlabel, vdata, vlabel, la, R_max=20):
    Ln = [la] * (len(tdata.shape) - 1)
    if len(tlabel.shape) > 2:
        Km = [la] * (len(tlabel.shape) - 1)
    else:
        Km = None
    test = HOPLS(R_max, Ln, Km)
    test.fit(tdata, tlabel)
    _, r, Q2 = test.predict(vdata, vlabel)
    return r, Q2


def do_testing(i, data_type, ss, X_mode, Y_mode, snr, lambda_max=10, R_max=20):
    PATH = "./"
    if data_type == "simple":
        data_type = ""
    elif data_type == "complex":
        data_type = "complex_"
    resname = PATH + f"results/res{i}_{data_type}s{ss}_X{X_mode}_Y{Y_mode}_{snr}dB.mat"
    if os.path.exists(resname):
        return
    filename = (
        PATH + f"dataset/data{i}_{data_type}s{ss}_X{X_mode}_Y{Y_mode}_{snr}dB.mat"
    )
    data = loadmat(filename)
    X = data["X"]
    Y = data["Y"]
    cv = KFold(n_folds)
    fold = 0
    PLS_r = []
    PLS_q2 = []
    HOPLS_l = []
    HOPLS_r = []
    HOPLS_q2 = []
    NPLS_r = []
    HOPLS_train_q2 = []
    PLS_train_q2 = []
    NPLS_train_q2 = []
    NPLS_q2 = []
    PLS_hyper = np.zeros((n_folds, R_max))
    HOPLS_hyper = np.zeros((n_folds, lambda_max - 1, R_max))
    NPLS_hyper = np.zeros((n_folds, R_max))
    for train_idx, valid_idx in cv.split(X, Y):
        X_train = torch.Tensor(X[train_idx])
        Y_train = torch.Tensor(Y[train_idx])
        X_valid = torch.Tensor(X[valid_idx])
        Y_valid = torch.Tensor(Y[valid_idx])

        results = []
        for R in range(1, R_max + 1):
            results.append(compute_q2_pls(X_train, Y_train, X_valid, Y_valid, R))
        old_Q2 = -np.inf
        PLS_hyper[fold] = results
        for i in range(len(results)):
            Q2 = results[i]
            if Q2 > old_Q2:
                best_r = i + 1
                old_Q2 = Q2
        PLS_q2s = compute_q2_pls(X_train, Y_train, X_train, Y_train, best_r)
        PLS_train_q2.append(PLS_q2s[-1])
        PLS_r.append(best_r)
        PLS_q2.append(old_Q2)

        results = []
        for lam in range(1, lambda_max):
            results.append(
                compute_q2_hopls(X_train, Y_train, X_valid, Y_valid, lam, R_max)
            )
        old_Q2 = -np.inf
        NPLS_hyper[fold] = results[0][1]
        for i in range(1, len(results)):
            r, Q2s = results[i]
            HOPLS_hyper[fold, i - 1] = Q2s
            Q2 = Q2s[r - 1]
            if Q2 > old_Q2:
                best_lam = i + 1
                best_r = r
                old_Q2 = Q2
        _, HOPLS_q2s = compute_q2_hopls(
            X_train, Y_train, X_train, Y_train, best_lam, best_r
        )
        _, NPLS_q2s = compute_q2_hopls(
            X_train, Y_train, X_train, Y_train, best_lam, best_r
        )
        HOPLS_train_q2.append(HOPLS_q2s[-1])
        NPLS_train_q2.append(NPLS_q2s[-1])
        HOPLS_l.append(best_lam)
        HOPLS_r.append(best_r)
        HOPLS_q2.append(old_Q2)
        best_npls_r = results[0][0]
        NPLS_r.append(best_npls_r)
        NPLS_q2.append(results[0][1][best_npls_r - 1])
        fold += 1
    results = {
        "PLS_R": PLS_r,
        "PLS_Q2": PLS_q2,
        "PLS_train": PLS_train_q2,
        "PLS_hyp": PLS_hyper,
        "HOPLS_R": HOPLS_r,
        "HOPLS_L": HOPLS_l,
        "HOPLS_Q2": HOPLS_q2,
        "HOPLS_train": HOPLS_train_q2,
        "HOPLS_hyp": HOPLS_hyper,
        "NPLS_R": NPLS_r,
        "NPLS_Q2": NPLS_q2,
        "NPLS_train": NPLS_train_q2,
        "NPLS_hyp": NPLS_hyper,
    }
    savemat(resname, results)


if __name__ == "__main__":
    data_type, ss, X_mode, Y_mode, snr = sys.argv[1:]
    # normalization = None  # 'remove_mean', 'zscore', 'normalize'
    # SNR = [5, 0, -2, -5]
    # modeY = [2, 3, 4]
    # modeX = [4, 5]
    n_dataset = 24
    # sample_size = [10, 20]
    n_folds = 5
    mat = []
    hyper = []
    Parallel(n_jobs=-1)(
        delayed(do_testing)(n, data_type, ss, X_mode, Y_mode, snr)
        for n in range(n_dataset)
    )
