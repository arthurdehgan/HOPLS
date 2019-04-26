import numpy as np
import tensorly as tl
from tensorly.tenalg import mode_dot, multi_mode_dot
from scipy.io import savemat


def generate(I1, In, Jm, X_mode, Y_mode=2, R=5, L=7, snr=10):
    T = tl.tensor(np.random.normal(size=(R, I1)))
    P = []
    for _ in range(X_mode - 1):
        P.append(tl.tensor(np.random.normal(size=(In, L)).T))
    G = tl.tensor(np.random.normal(size=[R] + [L] * (X_mode - 1)))
    Q = []
    for i in range(Y_mode - 1):
        Q.append(tl.tensor(np.random.normal(size=(Jm, L)).T))
    D = tl.tensor(np.random.normal(size=[R] + [L] * (Y_mode - 1)))
    E = tl.tensor(np.random.normal(size=[I1] + [In] * (X_mode - 1)))
    F = tl.tensor(np.random.normal(size=[I1] + [Jm] * (Y_mode - 1)))

    data = multi_mode_dot(G, [T] + P, np.arange(X_mode), transpose=True)
    target = multi_mode_dot(D, [T] + Q, np.arange(Y_mode), transpose=True)

    epsilon = 1 / (10 ** (snr / 10))
    data = data + epsilon * E
    target = target + epsilon * F

    return data, target


if __name__ == "__main__":
    for order in [3, 5]:
        for noise in [10, 5, 0, -5]:
            for modY in [2, 3, 5]:
                X, Y = generate(100, 10, 10, order, snr=noise)
                savemat(f"data_X{order}_Y{modY}_{noise}dB", {"X": X, "Y": Y})
