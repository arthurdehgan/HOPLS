import numpy as np
import tensorly as tl
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from scipy.io import loadmat, savemat
from hopls import qsquared, rmsep


if __name__ == "__main__":
    mat = []
    hyper = []
    for i, snr in enumerate([10, 5, 0, -5, -10]):
        data = loadmat(f"hox_data_{snr}dB.mat")
        og_X = tl.unfold(data["data"], 0)
        og_Y = tl.unfold(data["target"], 0)
        epsilon = 1 / (10 ** (snr / 10))
        this = []
        blob = []
        for _ in range(5):
            index = np.random.permutation(list(range(100)))
            X = og_X[index]
            Y = og_Y[index]

            # data = loadmat(f"ho_data_{snr}dB.mat")
            # X = data["data"]
            # Y = data["target"]

            X_train = X[:60]
            Y_train = Y[:60]
            # data = savemat(f"lo_data_{snr}dB_train.mat", {"X": X_train, "Y": Y_train})
            # data = savemat(f"ho_data_{snr}dB_train.mat", {"X": X_train, "Y": Y_train})
            X_valid = X[60:80]
            Y_valid = Y[60:80]
            # data = savemat(f"lo_data_{snr}dB_valid.mat", {"X": X_valid, "Y": Y_valid})
            # data = savemat(f"ho_data_{snr}dB_valid.mat", {"X": X_valid, "Y": Y_valid})
            X_test = X[80:100]
            Y_test = Y[80:100]
            # data = savemat(f"lo_data_{snr}dB_test.mat", {"X": X_test, "Y": Y_test})
            # data = savemat(f"ho_data_{snr}dB_test.mat", {"X": X_test, "Y": Y_test})
            # continue

            print("\n PLS SNR=" + str(snr) + "dB")
            old_Q2 = -np.inf
            for R in range(1, 20):
                test = PLSRegression(n_components=R)
                test.fit(X_train, Y_train)
                PLS_X_test = tl.unfold(X_valid, 0)
                PLS_Y_test = tl.unfold(Y_valid, 0)
                Y_pred = test.predict(X_valid)
                Q2 = qsquared(Y_valid, Y_pred)
                if Q2 > old_Q2:
                    best_params = {"R": R, "score": Q2, "pred": Y_pred}
                    old_Q2 = Q2

            Y_pred = test.predict(X_test)
            Q2 = qsquared(Y_test, Y_pred)
            R2 = r2_score(Y_test, Y_pred)
            RMSEP = r2_score(Y_test, Y_pred)

            # print("best param is R=" + str(best_params["R"]))
            # print("Q2: " + str(Q2))
            # print("R2: " + str(R2))
            # print("RMSEP: " + str(RMSEP))
            blob.append(best_params["R"])
            this.append(Q2)
        hyper.append(blob)
        mat.append(this)
    savemat(
        "PLS_res.mat", {"best_ncomp_test": np.array(hyper), "q2_test": np.asarray(mat)}
    )
