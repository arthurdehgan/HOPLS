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


def mmt(X,Y,N):
    if type(Y) is list:
        ndim = len(Y)
        for i in range(ndim):
            X = mmt(X,Y[i],i+1)
        Y = X
    else:
        n = N
        N = len(X.shape)
        sz = X.shape
        xp = set([x for x in range(N)])
        xp.remove(n)
        order = list([n]+ list(xp))
        newdata = X.permute((order))
        newdata = np.reshape(newdata,(sz[n],np.prod([sz[i] for i in list(xp)])),order='F')
        lm = np.tensordot(Y.transpose(0,1),newdata,(1,0))
        
        p = np.shape(Y)[1]
        newsz = [p] + [sz[i] for i in list(xp)]
        Y = np.reshape(lm,(newsz),order='F')
        Y = torch.Tensor(Y)
        inverse = [0] * len(order)
        for i, p in enumerate(order):
            inverse[p] = i
        iorder = inverse
        Y = Y.permute((iorder))
    return Y

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

def matricize(data):
    return np.reshape(data,(-1,np.prod([x for x in data.shape[1:]])),order='F')

def remove_mean(data):
    shape = tl.unfold(data[0],0).shape
    X_norm = torch.zeros(shape)
    for i in range(data.shape[0]):
        X_norm += tl.unfold(data[i],0)
    X_norm/=data.shape[0]
    
    for i in range(data.shape[0]):
#         data[i] -= X_norm.squeeze()
        data[i] -= np.reshape(X_norm,(data.shape[1:]))
    return data

# dat = loadmat("lo_data_-5dB.mat")
# og_X = torch.Tensor(dat["data"])
# og_Y = torch.Tensor(dat["target"])

dat = loadmat("data_X5_Y2_0dB.mat")
og_X = torch.Tensor(dat["X"])
og_Y = torch.Tensor(dat["Y"])

X = tl.unfold(og_X, 0)
Y = tl.unfold(og_Y, 0)
X_train = X[:60]
Y_train = Y[:60]
X_valid = X[60:80]
Y_valid = Y[60:80]
X_test = X[80:100]
Y_test = Y[80:100]

old_Q2 = -float("Inf")
for R in range(1, 10):
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

X = og_X[:60]
Y = og_Y[:60]

# for i in range(X.shape[0]):
#     X[i] -= torch.mean(X[i]) * torch.ones(X[i].shape)
#     Y[i] -= torch.mean(Y[i]) * torch.ones(Y[i].shape)


X = remove_mean(X)
Y = remove_mean(Y)

best_Q2 = 0
best_lambda = 0
for lam in range(1,10):
# lam = 3

    Ln = [lam] * (len(X.shape) - 1)
    modes = [x for x in range(len(X.shape))]
    R = 50
    In = X.shape
    N = len(Ln)
    M = Y.shape[-1]
    Er, Fr = X, Y
    P = tl.zeros((*X.shape[1:], R)).reshape(-1, R)
    W = tl.zeros((*X.shape[1:], R)).reshape(-1, R)
    Q = tl.zeros((M, R))
    T = tl.zeros((X.shape[0], R))
    D = torch.zeros((R, R))
    Gr, _ = tucker(Er, ranks=[1]+Ln)
    Gr = Gr
    # Beginning of the algorithm
    for r in range(R):
        # computing the covariance
        Cr = mode_dot(Er, Fr.transpose(0, 1), 0)
        # Cr = mmt(Er,Fr,0)
        # HOOI tucker decomposition of C
        _, latents = tucker(Cr,ranks=[1]+Ln)

        # Getting P and Q loadings
        qr = latents[0]
        Pr = latents[1:]
        # tr = mmt(Er,Pr,0)
        tr = multi_mode_dot(Er, Pr, list(range(1, len(Pr) + 1)), transpose=True)

        Gr_pi = pinv(matricize(Gr))
        
        tr = torch.mm(matricize(tr), Gr_pi)
        tr /= norm(tr)

        # recomposition of the core tensors
        ur = torch.mm(Fr, qr)
        dr = torch.mm(ur.transpose(0, 1), tr)
        D[r, r] = dr

        Pkron = kronecker([Pr[N - n - 1] for n in range(N)])
        W[:, r] = torch.mm(Pkron, Gr_pi).view(1, -1)
        Pr_ = torch.mm(matricize(Gr), Pkron.transpose(0, 1))
        P[:, r] = Pr_

        Q[:, r] = qr.view(-1)
        T[:, r] = tr.view(-1)
        X_hat = torch.mm(T, P.t())
        # X_hat = torch.mm(tr, Pr_)
        # Deflation
        Er = Er - np.reshape(X_hat, (Er.shape),order='F')
        Fr = Fr -(dr*torch.mm(tr, qr.transpose(0, 1)))


    Wfin = []
    for r in range(R):
        inter = pinv(torch.mm(P[:, : r + 1].transpose(0, 1), W[:, : r + 1]))
        W_star = torch.mm(W[:, : r + 1], inter)
        Wfin.append(
            torch.mm(W_star, torch.mm(D[: r + 1, : r + 1], Q[:, : r + 1].transpose(0, 1)))
        )
    best_q2_lambda =0
    best_r_lambda = 0
    for i in range(R):
        Y_pred = torch.mm(matricize(X), Wfin[i])
        Q2 = qsquared(Y, Y_pred)
        if Q2 > best_q2_lambda:
            best_q2_lambda = Q2
            best_r_lambda  = i+1
            # print(Q2)
        print(Q2,lam)
    # print()
    if best_q2_lambda>best_Q2:
        best_Q2 = best_q2_lambda
        best_r = best_r_lambda

        best_lambda = lam

print("HOPLS")
print("Q2: " + str(best_Q2))
print("Best R:"+str(best_r))
print("Best Lambda:" + str(best_lambda))
"""
PLot with modes on y and x and 3 curves of q2 scores for the 3 algorithms
"""
