from itertools import product
import numpy as np
import tensorly as tl
from tensorly.tenalg.n_mode_product import mode_dot
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error as mse
from hopls import HOPLS, qsquared, rmsep
from scipy.stats import zscore


if __name__ == "__main__":
    # Generation according to 4.1.1 of the paper equation (29)
    T = tl.tensor(np.random.normal(size=(10, 5)))
    P = tl.tensor(np.random.normal(size=(5, 10, 10)))
    Q = tl.tensor(np.random.normal(size=(5, 10, 10)))
    # Q = tl.tensor(np.random.normal(size=(5, 12)))
    E = tl.tensor(np.random.normal(size=(10, 10, 10)))
    F = tl.tensor(np.random.normal(size=(10, 10, 10)))
    X = mode_dot(P, T, 0)
    Y = mode_dot(Q, T, 0)

    # hyperparameters_search(X, Y, verbose=True)

    # see TABLE 1 for R and lambda values of PLS and HOPLS
    R_list = [5, 5, 3, 3]
    R_ho_list = [9, 7, 5, 3]
    lambda_list = [6, 5, 4, 5]
    for i, snr in enumerate([10, 5, 0, -5]):
        epsilon = 1 / (10 ** (snr / 10))
        noisy_X = X + epsilon * E
        noisy_Y = Y + epsilon * E

        print("\n PLS SNR=" + str(snr) + "dB")
        test = PLSRegression(n_components=R_list[i])
        PLS_X = tl.unfold(noisy_X, 0)
        PLS_Y = tl.unfold(noisy_Y, 0)
        test.fit(PLS_X, PLS_Y)
        Y_pred = test.predict(PLS_X)
        Y_pred = zscore(Y_pred)
        PLS_Y = zscore(PLS_Y)
        print("Q2: " + str(qsquared(PLS_Y, Y_pred)))
        print("R2: " + str(r2_score(PLS_Y, Y_pred)))
        print("RMSEP: " + str(np.mean(rmsep(PLS_Y, Y_pred))))

        print("\n HOPLS SNR=" + str(snr) + "dB")
        l = lambda_list[i]
        hop = HOPLS(R=R_ho_list[i], Ln=[l, l], Kn=[l, l])
        hopls = hop.fit(noisy_X, noisy_Y)
        Y_pred_hopls = hop.predict(noisy_X, noisy_Y)
        Y_pred_hopls = tl.unfold(zscore(Y_pred_hopls), 0)
        noisy_Y = tl.unfold(zscore(noisy_Y), 0)
        print("Q2: " + str(qsquared(noisy_Y, Y_pred_hopls)))
        print("R2: " + str(r2_score(noisy_Y, Y_pred_hopls)))
        print("RMSEP: " + str(np.mean(rmsep(noisy_Y, Y_pred_hopls))))

    # Q_err = Parallel(n_jobs=-1)(delayed(generate_and_test)() for _ in range(100))
    # print(np.mean(Q_err))
    # err = Parallel(n_jobs=-1)(delayed(generate_and_test)(rmsep) for _ in range(100))
    # print(np.mean(err))
    # r2err = Parallel(n_jobs=-1)(
    #     delayed(generate_and_test)(r2_score) for _ in range(100)
    # )
    # print(np.mean(r2err))
    # mserr = Parallel(n_jobs=-1)(delayed(generate_and_test)(mse) for _ in range(100))
    # print(np.mean(mserr))

    # for snr in [10, 5, 0, -5, -10]:
    #     print(f"SNR={snr}")
    #     epsilon = 1 / (10 ** (snr / 10))
    #     noisy_X = X + epsilon * E
    #     noisy_Y = Y  # + epsilon * F

    #     hyperparameters_search(noisy_X, noisy_Y)
