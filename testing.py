import torch
import numpy as np
import tensorly as tl
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.io import loadmat, savemat
from hopls import matricize, qsquared, HOPLS


if __name__ == "__main__":
    R_max = 20
    mat = []
    hyper = []
    for i, snr in enumerate([0]):
        filename = f"data_R5_L4_X5_Y2_{snr}dB.mat"
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

            old_Q2 = -np.inf
            for R in range(1, R_max):
                test = PLSRegression(n_components=R)
                test.fit(matricize(X_train), matricize(Y_train))
                Y_pred = test.predict(matricize(X_valid))
                Q2 = qsquared(matricize(Y_valid), matricize(Y_pred))
                if Q2 > old_Q2:
                    best_r = R
                    old_Q2 = Q2
            PLS_r.append(best_r)
            PLS_q2.append(old_Q2)

            old_Q2 = -np.inf
            for lam in range(1, 10):
                Ln = [lam] * (len(X.shape) - 1)
                test = HOPLS(R_max, Ln)
                test.fit(X_train, Y_train)
                Y_pred, r, Q2 = test.predict(X_valid, Y_valid)
                if Q2 > old_Q2:
                    best_lam = lam
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
