import numpy as np
import tensorly as tl
from tensorly.tenalg import mode_dot, multi_mode_dot
from scipy.io import savemat


def generate(I1, In, Jm, X_mode, Y_mode=2, R=5, L=7, snr=10):
    if isinstance(In, int):
        In = [In] * (X_mode - 1)
    if isinstance(Jm, int):
        Jm = [Jm] * (Y_mode - 1)
    T = tl.tensor(np.random.normal(size=(R, I1)))
    P = []
    for i in range(X_mode - 1):
        P.append(tl.tensor(np.random.normal(size=(In[i], L)).T))
    G = tl.tensor(np.random.normal(size=[R] + [L] * (X_mode - 1)))
    Q = []
    for i in range(Y_mode - 1):
        Q.append(tl.tensor(np.random.normal(size=(Jm[i], L)).T))
    D = tl.tensor(np.random.normal(size=[R] + [L] * (Y_mode - 1)))
    E = tl.tensor(np.random.normal(size=[I1] + In))
    F = tl.tensor(np.random.normal(size=[I1] + Jm))

    data = multi_mode_dot(G, [T] + P, np.arange(X_mode), transpose=True)
    target = multi_mode_dot(D, [T] + Q, np.arange(Y_mode), transpose=True)

    epsilon = 1 / (10 ** (snr / 10))
    data = data + epsilon * E
    target = target + epsilon * F

    return data, target


if __name__ == "__main__":
    R = 5
    L = 4
    In = [6, 7, 8]
    Jm = 9
    noise = 0
    X, Y = generate(20, In, Jm, len(In) + 1, Y_mode=2, R=R, L=L, snr=noise)
    savemat(f"data_R{R}_L{L}_X4_Y2_{noise}dB", {"X": X, "Y": Y})
    # noise = 0
    # modeY = 2
    # R = 5
    # L = 4
    # for modeX in [3, 4, 5, 6]:
    #     X, Y = generate(20, 10, 10, modeX, Y_mode=modeY, R=R, L=L, snr=noise)
    #     savemat(f"data_R{R}_L{L}_X{modeX}_Y{modeY}_{noise}dB", {"X": X, "Y": Y})
