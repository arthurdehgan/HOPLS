import numpy as np
import tensorly as tl
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.io import loadmat, savemat
from hopls_numpy import matricize, qsquared, HOPLS
from joblib import Parallel, delayed


def compute_q2_hopls(tdata, tlabel, vdata, vlabel, la):
    Ln = [la] * (len(X.shape) - 1)
    Km = [la] * (len(Y.shape) - 1)
    test = HOPLS(50, Ln, Km)
    test.fit(tdata, tlabel)
    _, r, Q2 = test.predict(vdata, vlabel)
    return r, Q2


if __name__ == "__main__":
    R_max = 20
    mat = []
    hyper = []

    for i, snr in enumerate([0]):
        filename = f"dataset/data_s20_X3_Y3_0dB.mat"
        print(filename)
        data = loadmat(filename)
        X = data["X"]
        Y = data["Y"]
        cv = KFold(2)
        PLS_r = []
        PLS_q2 = []
        HOPLS_l = []
        HOPLS_r = []
        HOPLS_q2 = []
        for train_idx, valid_idx in cv.split(X, Y):
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_valid = X[valid_idx]
            Y_valid = Y[valid_idx]

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

            results = Parallel(n_jobs=-1)(
                delayed(compute_q2_hopls)(X_train, Y_train, X_valid, Y_valid, lam)
                for lam in range(1, 10)
            )
            old_Q2 = -np.inf
            for i in range(8):
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
        # print(
        #     np.mean(HOPLS_q2[:5]),
        #     np.mean(HOPLS_q2[5:10]),
        #     np.mean(HOPLS_q2[10:15]),
        #     np.mean(HOPLS_q2[15:20]),
        # )
    # savemat(
    #     "PLS_res.mat", {"best_ncomp_test": np.array(hyper), "q2_test": np.asarray(mat)}
    # )
