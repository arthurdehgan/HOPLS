import torch
import numpy as np
import tensorly as tl
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.io import loadmat, savemat
from hopls import matricize, qsquared, HOPLS
from joblib import Parallel, delayed


def compute_q2_pls(tdata, tlabel, vdata, vlabel, Rval):
    test = PLSRegression(n_components=Rval)
    test.fit(matricize(tdata), matricize(tlabel))
    Y_pred = test.predict(matricize(vdata))
    Q2 = qsquared(matricize(vlabel), matricize(Y_pred))
    return Q2


def compute_q2_hopls(tdata, tlabel, vdata, vlabel, la):
    Ln = [la] * (len(X.shape) - 1)
    test = HOPLS(50, Ln)
    test.fit(tdata, tlabel)
    _, r, Q2 = test.predict(vdata, vlabel)
    return r, Q2


if __name__ == "__main__":
    mat = []
    hyper = []
    for i, snr in enumerate([10, 5, 0, -5]):
        filename = f"../datasets/data_X5_Y2_{snr}dB.mat"
        print(filename)
        data = loadmat(filename)
        X = data["X"]
        Y = data["Y"]
        cv = KFold(5)
        PLS_r = []
        PLS_q2 = []
        HOPLS_l = []
        HOPLS_r = []
        HOPLS_q2 = []
        for train_idx, valid_idx in cv.split(X, Y):
            X_train = torch.Tensor(X[train_idx])
            Y_train = torch.Tensor(Y[train_idx])
            X_valid = torch.Tensor(X[valid_idx])
            Y_valid = torch.Tensor(Y[valid_idx])

            results = Parallel(n_jobs=-1)(
                delayed(compute_q2_pls)(X_train, Y_train, X_valid, Y_valid, R)
                for R in range(1, 50)
            )
            old_Q2 = -np.inf
            for i in range(49):
                Q2 = results[i]
                if Q2 > old_Q2:
                    best_r = i + 1
                    old_Q2 = Q2
            PLS_r.append(best_r)
            PLS_q2.append(old_Q2)

            results = Parallel(n_jobs=-1)(
                delayed(compute_q2_hopls)(X_train, Y_train, X_valid, Y_valid, lam)
                for lam in range(1, 10)
            )
            old_Q2 = -np.inf
            for i in range(9):
                r, Q2 = results[i]
                if Q2 > old_Q2:
                    best_lam = i + 1
                    best_r = r
                    old_Q2 = Q2

            # print("best param is R=" + str(best_params["R"]))
            # print("Q2: " + str(Q2))
            HOPLS_l.append(best_lam)
            HOPLS_r.append(best_r)
            HOPLS_q2.append(old_Q2)
        # hyper.append(PLS_r)
        # mat.append(PLS_q2)
        print("PLS")
        print(PLS_r, np.mean(PLS_q2))
        print("HOPLS")
        print(HOPLS_r)
        print(HOPLS_l)
        print(np.mean(HOPLS_q2))
    # savemat(
    #     "PLS_res.mat", {"best_ncomp_test": np.array(hyper), "q2_test": np.asarray(mat)}
    # )
