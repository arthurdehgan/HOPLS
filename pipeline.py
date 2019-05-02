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
    if data_type == "simple":
        data_type = ""
    elif data_type == "complex":
        data_type = "complex_"
    filename = f"dataset/data{i}_{data_type}s{ss}_X{X_mode}_Y{Y_mode}_{snr}dB.mat"
    data = loadmat(filename)
    X = data["X"]
    Y = data["Y"]
    cv = KFold(n_folds)
    PLS_r = []
    PLS_q2 = []
    HOPLS_l = []
    HOPLS_r = []
    HOPLS_q2 = []
    NPLS_r = []
    NPLS_q2 = []
    for train_idx, valid_idx in cv.split(X, Y):
        X_train = torch.Tensor(X[train_idx])
        Y_train = torch.Tensor(Y[train_idx])
        X_valid = torch.Tensor(X[valid_idx])
        Y_valid = torch.Tensor(Y[valid_idx])

        results = []
        for R in range(1, R_max):
            results.append(compute_q2_pls(X_train, Y_train, X_valid, Y_valid, R))
        old_Q2 = -np.inf
        for i in range(len(results)):
            Q2 = results[i]
            if Q2 > old_Q2:
                best_r = i + 1
                old_Q2 = Q2
        PLS_r.append(best_r)
        PLS_q2.append(old_Q2)

        results = []
        for lam in range(1, lambda_max):
            results.append(
                compute_q2_hopls(X_train, Y_train, X_valid, Y_valid, lam, R_max)
            )
        old_Q2 = -np.inf
        for i in range(1, len(results)):
            r, Q2 = results[i]
            if Q2 > old_Q2:
                best_lam = i + 1
                best_r = r
                old_Q2 = Q2
        HOPLS_l.append(best_lam)
        HOPLS_r.append(best_r)
        HOPLS_q2.append(old_Q2)
        NPLS_r.append(results[0][0])
        NPLS_q2.append(results[0][1])
    filename = f"results/res{i}_{data_type}s{ss}_X{X_mode}_Y{Y_mode}_{snr}dB.mat"
    results = {
        "PLS_R": PLS_r,
        "PLS_Q2": PLS_q2,
        "HOPLS_R": HOPLS_r,
        "HOPLS_L": HOPLS_l,
        "HOPLS_Q2": HOPLS_q2,
        "NPLS_R": NPLS_r,
        "NPLS_Q2": NPLS_q2,
    }
    savemat(filename, results)


if __name__ == "__main__":
    ARGS = sys.argv[1:]
    data_type = ARGS[0]
    ss, X_mode, Y_mode, snr = list(map(int, ARGS[1:]))
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
