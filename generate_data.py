import numpy as np
import tensorly as tl
from tensorly.tenalg import mode_dot, multi_mode_dot
from scipy.io import savemat


def generate(I1, In, Jm, X_mode=None, Y_mode=None, R=5, snr=10):
    if isinstance(In, tuple) or isinstance(In, list):
        X_mode = len(In) + 1
    elif X_mode is None:
        assert 1, "X_mode should not be set to None if In is an int"
    else:
        In = (X_mode - 1) * [In]
    X_shape = [I1] + list(In)

    if isinstance(Jm, tuple) or isinstance(Jm, list):
        Y_mode = len(Jm) + 1
    elif X_mode is None:
        assert 1, "Y_mode should not be set to None if Jn is an int"
    else:
        Jm = (Y_mode - 1) * [Jm]
    Y_shape = [I1] + list(Jm)

    T = np.random.normal(size=(I1, R))
    G = np.random.normal(size=(np.prod(In), R))
    E = np.random.normal(size=(I1, np.prod(In)))
    data = np.matmul(T, G.T)

    D = np.random.normal(size=(np.prod(Jm), R))
    F = np.random.normal(size=(I1, np.prod(Jm)))
    target = np.matmul(T, D.T)

    epsilon = 1 / (10 ** (snr / 10))
    data = data + epsilon * E
    target = target + epsilon * F

    return data.reshape(X_shape, order="F"), target.reshape(Y_shape, order="F")


def generate_complex(I1, In, Jm, X_mode=None, Y_mode=2, R=5, L=7, snr=10):
    if isinstance(In, tuple) or isinstance(In, list):
        X_mode = len(In) + 1
    elif X_mode is None:
        assert 1, "X_mode should not be set to None if In is an int"
    else:
        In = (X_mode - 1) * [In]

    if isinstance(Jm, tuple) or isinstance(Jm, list):
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
    # I1 = 20
    # In = (10, 10)
    # Jm = (10, 10)
    # snr = 0
    # X, Y = generate(10, In, Jm, R=5, snr=snr)
    # savemat(f"dataset/data_s{I1}_X{len(In)+1}_Y{len(Jm)+1}_{snr}dB", {"X": X, "Y": Y})
    for i in range(24):
        for snr in [5, 0, -2, -5]:
            for In in [[20, 10], [20, 10, 20, 10]]:
                for Jm in [[10], [10, 5]]:
                    for ss in [10, 20]:
                        X, Y = generate_complex(ss, In, Jm, R=5, snr=snr)
                        savemat(
                            f"dataset/data{i}_complex_s{ss}_X{len(In)+1}_Y{len(Jm)+1}_{snr}dB",
                            {"X": X, "Y": Y},
                        )
