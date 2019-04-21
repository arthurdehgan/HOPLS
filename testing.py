from itertools import product
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg.n_mode_product import mode_dot
from tensorly.tenalg import kronecker
import numpy as np
from numpy.linalg import svd, pinv
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression


if __name__ == "__main__":
    # arbitrarly chosen epsilon
    epsilon = 5e-4
    # Generation according to 4.1.1 of the paper equation (29)
    T = tl.tensor(np.random.normal(size=(20, 5)))
    P = tl.tensor(np.random.normal(size=(5, 10, 10)))
    Q = tl.tensor(np.random.normal(size=(5, 10, 10)))
    E = tl.tensor(np.random.normal(size=(20, 10, 10)))
    F = tl.tensor(np.random.normal(size=(20, 10, 10)))
    X = mode_dot(P, T, 0)  # + epsilon * E
    Y = mode_dot(Q, T, 0)  # + epsilon * F
    old_mse = np.inf

    for snr in [10, 5, 0, -5, -10]:
        print(f"SNR={snr}")
        epsilon = 1 / (10 ** (snr / 10))
        noisy_X = X + epsilon * E
        noisy_Y = Y + epsilon * F

        # Just like in the paper, we simplify by having l for all ranks
        for R, l in product(range(3, 7), range(3, 10)):
            hopls = HOPLS(R)
            rmsep = np.mean(hopls.score(noisy_X, noisy_Y, [l, l], [l, l]))
            if rmsep < old_mse:
                old_mse = rmsep
                best_params = [R, l, rmsep]
            print(rmsep)
        print("Best model is with R={} and l={}, rmsep={:.2f}".format(*best_params))
