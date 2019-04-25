from itertools import product
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import multi_mode_dot, kronecker, mode_dot
from sklearn.cross_decomposition import PLSRegression
from scipy.io import loadmat
import torch
from torch import norm, svd, pinverse as pinv

tl.set_backend("pytorch")


def cov(A, B):
    C = torch.zeros(*(list(A.shape[1:]) + list(B.shape[1:])))
    dim = len(A.shape[1:])
    for idx in product(*[range(C.shape[i]) for i in range(len(C.shape))]):
        for i1 in range(A.shape[0]):
            X_index = tuple([i1] + list(idx[:dim]))
            Y_index = tuple([i1] + list(idx[dim:]))
            C[idx] += X[X_index] * B[Y_index]
    return C


def qsquared(y_true, y_pred):
    """Compute the Q^2 Error between two arrays."""
    return 1 - ((norm(y_true - y_pred) ** 2) / (norm(y_true) ** 2))


snr = 10
T = tl.tensor(np.random.normal(size=(100, 5)))
P = tl.tensor(np.random.normal(size=(5, 10, 10)))
Q = tl.tensor(np.random.normal(size=(5, 5)))
E = tl.tensor(np.random.normal(size=(100, 10, 10)))
F = tl.tensor(np.random.normal(size=(100, 5)))
X = mode_dot(P, T, 0)
Y = mode_dot(Q, T, 0)
epsilon = 1 / (10 ** (snr / 10))
noisy_X = X + epsilon * E
noisy_Y = Y + epsilon * F
og_X = torch.Tensor(noisy_X)
og_Y = torch.Tensor(noisy_Y)

# X = tl.unfold(og_X, 0)
# Y = tl.unfold(og_Y, 0)
# X_train = X[:60]
# Y_train = Y[:60]
# X_valid = X[60:80]
# Y_valid = Y[60:80]
# X_test = X[80:100]
# Y_test = Y[80:100]

# old_Q2 = -float("Inf")
# for R in range(1, 20):
#     test = PLSRegression(n_components=R)
#     test.fit(X_train, Y_train)
#     PLS_X_test = tl.unfold(X_valid, 0)
#     PLS_Y_test = tl.unfold(Y_valid, 0)
#     Y_pred = torch.Tensor(test.predict(X_valid))
#     Q2 = qsquared(Y_valid, Y_pred)
#     if Q2 > old_Q2:
#         best_params = {"R": R, "score": Q2, "pred": Y_pred}
#         old_Q2 = Q2
#
# print("PLS sanity check")
# print("best param is R=" + str(best_params["R"]))
# print("Q2: " + str(float(Q2)))

X = og_X[:60]
Y = og_Y[:60]

for i in range(X.shape[0]):
    X[i] -= torch.mean(X[i]) * torch.ones(X[i].shape)
    Y[i] -= torch.mean(Y[i]) * torch.ones(Y[i].shape)

Ln = [2] * (len(X.shape) - 1)

R = 5
In = X.shape
N = len(Ln)
M = Y.shape[-1]
Er, Fr = X, Y
P = tl.zeros((*X.shape[1:], R)).reshape(-1, R)
W = tl.zeros((*X.shape[1:], R)).reshape(-1, R)
Q = tl.zeros((M, R))
T = tl.zeros((X.shape[0], R))
D = torch.zeros((R, R))
Gr, _ = tucker(Er, ranks=[1] + Ln)

# Beginning of the algorithm
for r in range(R):
    # computing the covariance
    Cr = mode_dot(Er, Fr.transpose(0, 1), 0)

    # HOOI tucker decomposition of C
    _, latents = tucker(Cr, ranks=[1] + Ln)

    # Getting P and Q loadings
    qr = latents[0]
    Pr = latents[1:]
    tr = multi_mode_dot(Er, Pr, list(range(1, len(Pr) + 1)), transpose=True)
    Gr_pi = pinv(tl.unfold(Gr, 0))
    tr = torch.mm(tl.unfold(tr, 0), Gr_pi)
    tr /= norm(tr)

    # recomposition of the core tensors
    ur = torch.mm(Fr, qr)
    dr = torch.mm(ur.transpose(0, 1), tr)
    D[r, r] = dr

    Pkron = kronecker([Pr[N - n - 1] for n in range(N)])
    W[:, r] = torch.mm(Pkron, Gr_pi).view(1, -1)
    Pr = torch.mm(tl.unfold(Gr, 0), Pkron.transpose(0, 1))
    P[:, r] = Pr

    Q[:, r] = qr.view(-1)
    T[:, r] = tr.view(-1)
    X_hat = torch.mm(tr, Pr)

    # Deflation
    Er = Er - X_hat.view(Er.shape)
    Fr = Fr - dr * torch.mm(tr, qr.transpose(0, 1))


Wfin = []
for r in range(R):
    inter = pinv(torch.mm(P[:, : r + 1].transpose(0, 1), W[:, : r + 1]))
    W_star = torch.mm(W[:, : r + 1], inter)
    Wfin.append(
        torch.mm(W_star, torch.mm(D[: r + 1, : r + 1], Q[:, : r + 1].transpose(0, 1)))
    )

Y_pred = torch.mm(tl.unfold(X, 0), Wfin[-1])

Q2 = qsquared(Y, Y_pred)
print("HOPLS")
print("Q2: " + str(Q2))
"""
PLot with modes on y and x and 3 curves of q2 scores for the 3 algorithms
"""
