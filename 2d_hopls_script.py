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
    return np.reshape(data,(-1,int(np.prod([x for x in data.shape[1:]]))),order='F')
import torch.nn.functional as f
def remove_mean(data):
    X_mean = data.mean(dim=0)
    data  -= X_mean

#     X_std = data.std(dim=0)
#     X_std[X_std ==0.0]=1.0
#     data /= X_std 

    # data = f.normalize(data,dim=0,p=2)
    return data    

# dat = loadmat("../datasets/hox_data_-5dB.mat")
# og_X = torch.Tensor(dat["data"])
# og_Y = torch.Tensor(dat["target"])

dat = loadmat("../datasets/testing_data.mat")
og_X = torch.Tensor(dat["X"])
og_Y = torch.Tensor(dat["Y"])

X = tl.unfold(og_X,0)
Y = tl.unfold(og_Y,0)

print(np.shape(og_X),np.shape(og_Y))
X_train = X[:16]
Y_train = Y[:16]
X_valid = X[16:20]
Y_valid = Y[16:20]
# X_test = X[80:100]
# Y_test = Y[80:100]

old_Q2 = -float("Inf")
for R in range(1, 10):
    test = PLSRegression(n_components=R)
    test.fit(X_train, Y_train)
    Y_pred = torch.Tensor(test.predict(X_valid))
    Q2 = qsquared(Y_valid, Y_pred)
    if Q2 > old_Q2:
        best_params = {"R": R, "score": Q2, "pred": Y_pred}
        old_Q2 = Q2

print("PLS sanity check")
print("best param is R=" + str(best_params["R"]))
print("Q2: " + str(float(Q2)))


X = remove_mean(og_X)
Y = remove_mean(og_Y)

X_train = X[:16]
Y_train = Y[:16]
X_valid = X[16:20]
Y_valid = Y[16:20]

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
    Er, Fr = X_train, Y_train
    T, P = [],[]

    W = []
    Q = []
    D = torch.Tensor(np.identity(R))
    Gr, _ = tucker(Er, ranks=[1]+Ln)
    Gr = Gr
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
        Gr_pi = torch.pinverse(matricize(Gr))
        tr = torch.mm(matricize(tr), Gr_pi)
        tr /= torch.norm(tr)
        T.append(tr)

        # recomposition of the core tensor of Y
        ur = torch.mm(Fr, qr)
        dr = torch.mm(ur.t(), tr)
        D[r, r] = dr
        Pkron = kronecker([Pr[N - n - 1] for n in range(N)])
        W.append(torch.mm(Pkron, Gr_pi))
        
        P.append(torch.mm(matricize(Gr), Pkron.t()).t())

        # Gathering of R variables
        Q.append(qr)
        D[r, r] = dr
        # Deflation
        X_hat = torch.mm(torch.cat(T,dim=1), torch.cat(P,dim=1).t())
        Er = X_train - np.reshape(X_hat, (X_train.shape), order="F")
        Fr = Fr - dr * torch.mm(tr, qr.t())

    Q = torch.cat(Q,dim=1)
    T = torch.cat(T,dim=1)
    P = torch.cat(P,dim=1)
    W = torch.cat(W,dim=1)


    Wfin = []
    for r in range(R):
        inter = torch.pinverse(
                torch.mm(P[:,: r + 1].t(), W[:, : r + 1])
                + 2e-8 * torch.diag(torch.rand(r + 1))
            )
        W_star = torch.mm(W[:, : r + 1], inter)
        Wfin.append(
            torch.mm(
                W_star, torch.mm(D[: r + 1, : r + 1], Q[:, : r + 1].transpose(0, 1))
            )
        )
    best_q2_lambda =0
    best_r_lambda = 0


    for i in range(R):
        Y_pred = torch.mm(matricize(X_valid), Wfin[i])
        # Y_pred, _ = remove_mean(Y_pred, self.mY)
        Q2 = qsquared(Y_valid, Y_pred)
        if Q2 > best_q2_lambda:
            best_q2_lambda = Q2
            best_r_lambda  = i+1
        print(Q2,lam)
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
