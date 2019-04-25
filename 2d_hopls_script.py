from itertools import product
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
data = loadmat(f"hox_data_{snr}dB.mat")
og_X = torch.Tensor(data["data"])
og_Y = torch.Tensor(data["target"])

X = tl.unfold(og_X, 0)
Y = tl.unfold(og_Y, 0)
X_train = X[:60]
Y_train = Y[:60]
X_valid = X[60:80]
Y_valid = Y[60:80]
X_test = X[80:100]
Y_test = Y[80:100]

old_Q2 = -float("Inf")
for R in range(1, 20):
    test = PLSRegression(n_components=R)
    test.fit(X_train, Y_train)
    PLS_X_test = tl.unfold(X_valid, 0)
    PLS_Y_test = tl.unfold(Y_valid, 0)
    Y_pred = torch.Tensor(test.predict(X_valid))
    Q2 = qsquared(Y_valid, Y_pred)
    if Q2 > old_Q2:
        best_params = {"R": R, "score": Q2, "pred": Y_pred}
        old_Q2 = Q2

print("PLS sanity check")
print("best param is R=" + str(best_params["R"]))
print("Q2: " + str(float(Q2)))

X = og_X
Y = og_Y

Ln = [5] * (len(X.shape) - 1)

In = X.shape
M = Y.shape[-1]
Er, Fr = X, Y
P = []
Q = tl.zeros((M, R))
G = tl.zeros((R, *Ln))
D = tl.zeros((R, R))
T = tl.zeros((In[0], R))

# Beginning of the algorithm
for r in range(R):
    # computing the covariance
    Cr = mode_dot(Er, Fr.transpose(0, 1), 0)

    # HOOI tucker decomposition of C
    Gr_C, latents = tucker(Cr, ranks=[1] + Ln)

    # Getting P and Q loadings
    qr = latents[0]
    Pr = latents[1:]
    tr = multi_mode_dot(Er, Pr, list(range(1, len(Pr) + 1)), transpose=True)
    tr = torch.mm(tl.unfold(tr, 0), pinv(Gr_C.reshape(1, -1)))
    tr /= norm(tr)

    # recomposition of the core tensors
    Gr = tl.tucker_to_tensor(Er, [tr] + Pr, transpose_factors=True)
    ur = torch.mm(Fr, qr)
    dr = torch.mm(ur.transpose(0, 1), tr)

    # Gathering of R variables
    P.append(Pr)
    Q[:, r] = qr.transpose(0, 1)
    G[r] = Gr[0]
    D[r, r] = dr
    T[:, r] = tr[:, 0]

    # Deflation
    Er = Er - tl.tucker_to_tensor(Gr, [tr] + Pr)
    Fr = Fr - dr * torch.mm(tr, qr.transpose(0, 1))

model = (P, Q, G, D, T)
P, Q, G, D, T = model
N = len(Ln)

Q_star = tl.zeros(Q.shape)
W = tl.zeros((*X.shape[1:], R)).reshape(-1, R)

for r in range(R):
    G_pi = pinv(tl.unfold(G[r], 0))
    W[:, r] = torch.mm(
        kronecker([P[r][N - n - 1] for n in range(N)]), G_pi.view(-1, 1)
    ).view(-1)
    Q_star[:, r] = D[r, r] * Q[:, r]

Y_pred = torch.mm(torch.mm(tl.unfold(X, 0), W), Q_star.transpose(0, 1)).reshape(Y.shape)

Q2 = qsquared(Y, Y_pred)
print("HOPLS")
print("Q2: " + str(Q2))
"""
PLot with modes on y and x and 3 curves of q2 scores for the 3 algorithms
"""
