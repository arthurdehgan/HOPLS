from itertools import product
import numpy as np
import tensorly as tl
from tensorly.tenalg.n_mode_product import mode_dot
from sklearn.cross_decomposition import PLSRegression
from hopls import HOPLS, qsquared


def hyper_params_search(data, target, metric=None, verbose=True):
    old_error = -np.inf
    # Just like in the paper, we simplify by having l for all ranks
    for R, l in product(range(3, 7), range(3, 10)):
        hopls = HOPLS(
            R, [l] * (len(data.shape) - 1), [l] * (len(data.shape) - 1), metric=None
        )
        error = np.mean(hopls.score(data, target))
        if error > old_error:
            old_error = error
            best_params = [R, l, error]
    if verbose:
        print("Best model is with R={} and l={}, error={:.2f}".format(*best_params))
    return best_params


if __name__ == "__main__":
    # Generation according to 4.1.1 of the paper equation (29)
    T = tl.tensor(np.random.normal(size=(20, 5)))
    P = tl.tensor(np.random.normal(size=(5, 10, 10)))
    Q = tl.tensor(np.random.normal(size=(5, 10, 10)))
    E = tl.tensor(np.random.normal(size=(20, 10, 10)))
    F = tl.tensor(np.random.normal(size=(20, 10, 10)))
    X = mode_dot(P, T, 0)
    Y = mode_dot(Q, T, 0)

    hyper_params_search(X, Y, verbose=True)

    # test = PLSRegression(n_components=5)
    # test.fit(X, Y)
    # print(qsquared(Y, test.predict(X)))

    # for _ in range(100):
    #     _, _, err = hyper_params_search(X, X, verbose=False)
    # print(err / 100)

    # for snr in [10, 5, 0, -5, -10]:
    #     print(f"SNR={snr}")
    #     epsilon = 1 / (10 ** (snr / 10))
    #     noisy_X = X + epsilon * E
    #     noisy_Y = Y  # + epsilon * F

    #     hyper_params_search(noisy_X, noisy_Y)
