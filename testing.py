from itertools import product
import numpy as np
import tensorly as tl
from tensorly.tenalg.n_mode_product import mode_dot
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error as mse
from joblib import Parallel, delayed
from hopls import HOPLS, qsquared, rmsep


def hyperparameters_search(data, target, metric=None, verbose=True):
    old_error = -np.inf
    # old_error = np.inf
    # Just like in the paper, we simplify by having l for all ranks
    for R, l in product(range(3, 7), range(3, 20)):
        hopls = HOPLS(
            R, [l] * (len(data.shape) - 1), [l] * (len(target.shape) - 1), metric=metric
        )
        error = np.mean(hopls.score(data, target))
        if error > old_error:
            # if error < old_error:
            old_error = error
            best_params = [R, l, error]
    if verbose:
        print("Best model is with R={} and l={}, error={:.2f}".format(*best_params))
    return best_params


def generate_and_test(metric=qsquared):
    T = tl.tensor(np.random.normal(size=(20, 5)))
    P = tl.tensor(np.random.normal(size=(5, 10, 11)))
    Q = tl.tensor(np.random.normal(size=(5, 12, 13)))
    X = mode_dot(P, T, 0)
    Y = mode_dot(Q, T, 0)
    l = 4
    Ln = [l] * (len(X.shape) - 1)
    Kn = None
    if len(Y.shape) > 2:
        Kn = [l] * (len(Y.shape) - 1)
    model = HOPLS(5, Ln, Kn)
    model.fit(X, Y)
    Y_pred = model.predict(X, Y)
    return metric(Y, Y_pred)


if __name__ == "__main__":
    # Generation according to 4.1.1 of the paper equation (29)
    T = tl.tensor(np.random.normal(size=(10, 5)))
    P = tl.tensor(np.random.normal(size=(5, 11, 12, 13)))
    Q = tl.tensor(np.random.normal(size=(5, 6, 7)))
    # Q = tl.tensor(np.random.normal(size=(5, 12)))
    E = tl.tensor(np.random.normal(size=(10, 10, 11)))
    F = tl.tensor(np.random.normal(size=(10, 10, 10)))
    X = mode_dot(P, T, 0)
    Y = mode_dot(Q, T, 0)

    # hyperparameters_search(X, Y, verbose=True)

    test = PLSRegression(n_components=5)
    PLS_X = X.reshape(X.shape[0], -1)  # + 100 * E.reshape(X.shape[0], -1)
    PLS_Y = Y.reshape(X.shape[0], -1)
    test.fit(PLS_X, PLS_Y)
    Y_pred = test.predict(PLS_X)
    print(qsquared(PLS_Y, Y_pred))
    print(np.mean(rmsep(PLS_Y, Y_pred)))
    print(r2_score(PLS_Y, Y_pred))
    print(mse(PLS_Y, Y_pred))

    for l in range(3, 6):
        hop = HOPLS(R=5, Ln=[l, l + 1, l + 2], Kn=[l, l + 1])
        hop.fit(X, Y)
        print(hop.score(X, Y))

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
