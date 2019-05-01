import numpy as np
import tensorly as tl
from tensorly.tenalg import mode_dot, multi_mode_dot
from scipy.io import savemat


def generate(I1, In, Jm, X_mode=None, Y_mode=2, R=5, L=7, snr=10):
    if type(In) == tuple or type(In) == list:
        X_mode = len(In) + 1
    elif X_mode is None:
        assert 1, "X_mode should not be set to None if In is an int"
    else:
        In = (X_mode - 1) * [In]

    if type(Jm) == tuple or type(Jm) == list:
        Y_mode = len(Jm) + 1
    elif X_mode is None:
        assert 1, "Y_mode should not be set to None if Jn is an int"
    else:
        Jm = (Y_mode - 1) * [Jm]

    T = tl.tensor(np.random.normal(size=(R, I1)))
    P, Q = [], []
    for i in range(X_mode - 1):
        P.append(tl.tensor(np.random.normal(size=(In[i], L)).T))
    for i in range(Y_mode - 1):
        Q.append(tl.tensor(np.random.normal(size=(Jm[i], L)).T))

    E = tl.tensor(np.random.normal(size=[I1] + list(In)))
    F = tl.tensor(np.random.normal(size=[I1] + list(Jm)))
    D = tl.tensor(np.random.normal(size=[R] + [L] * (Y_mode - 1)))
    G = tl.tensor(np.random.normal(size=[R] + [L] * (X_mode - 1)))

    data = multi_mode_dot(G, [T] + P, np.arange(X_mode), transpose=True)
    target = multi_mode_dot(D, [T] + Q, np.arange(Y_mode), transpose=True)

    epsilon = 1 / (10 ** (snr / 10))
    data = data + epsilon * E
    target = target + epsilon * F

    return data, target


if __name__ == "__main__":
    X, Y = generate(100, (7, 8, 9), (10), 4, snr=0)
    savemat(f"testing_data", {"X": X, "Y": Y})
#     for order in [3, 5]:
#         for noise in [10, 5, 0, -5]:
#             for modY in [2, 3, 5]:
#                 X, Y = generate(100, 10, 10, order, snr=noise)
#                 savemat(f"data_X{order}_Y{modY}_{noise}dB", {"X": X, "Y": Y})

#
