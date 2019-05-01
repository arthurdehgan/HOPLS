import torch
import numpy as np
import tensorly as tl
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.io import loadmat, savemat
from hopls import matricize, qsquared, HOPLS
from joblib import Parallel, delayed


def compute_q2_hopls(tdata, tlabel, vdata, vlabel, la):
    Ln = [la] * (len(X.shape) - 1)
    test = HOPLS(50, Ln)
    test.fit(tdata, tlabel)
    _, r, Q2 = test.predict(vdata, vlabel)
    return r, Q2


def normalize1(data):
    data -= data.mean(dim=0)
    return data


def normalize2(data):
    data -= data.mean(dim=0)
    data /= data.std(dim=0, unbiased=False)
    return data


def normalize3(data):
    return torch.nn.functional.normalize(data, dim=0, p=2)


if __name__ == "__main__":
    R_max = 20
    normal_list = [normalize1, normalize2, normalize3]
    mat = []
    hyper = []

    for i, snr in enumerate([0]):
        filename = f"testing_data.mat"
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

            # old_Q2 = -np.inf
            # for R in range(1, R_max):
            #     test = PLSRegression(n_components=R)
            #     test.fit(matricize(X_train), matricize(Y_train))
            #     Y_pred = test.predict(matricize(X_valid))
            #     Q2 = qsquared(matricize(Y_valid), matricize(Y_pred))
            #     if Q2 > old_Q2:
            #         best_r = R
            #         old_Q2 = Q2
            # PLS_r.append(best_r)
            # PLS_q2.append(old_Q2)

            for norm in normal_list:
                X_train = norm(X_train)
                X_valid = norm(X_valid)
                Y_train = norm(Y_train)
                Y_valid = norm(Y_valid)
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
        print([float(a) for a in HOPLS_q2])
        print(np.mean(HOPLS_q2[:5]), np.mean(HOPLS_q2[5:10]), np.mean(HOPLS_q2[10:15]))
    # savemat(
    #     "PLS_res.mat", {"best_ncomp_test": np.array(hyper), "q2_test": np.asarray(mat)}
    # )
