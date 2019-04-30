from itertools import product
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import mode_dot, multi_mode_dot, kronecker
import numpy as np
import torch

tl.set_backend("pytorch")
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


def matricize(data):
    return torch.Tensor(
        np.reshape(data, (-1, np.prod([x for x in data.shape[1:]])), order="F")
    )


def remove_mean(data, X_norm=None):
    if X_norm is None:
        shape = tl.unfold(data[0], 0).shape
        X_norm = torch.zeros(shape)
        for i in range(data.shape[0]):
            X_norm += matricize(data[i])
        X_norm /= data.shape[0]
    for i in range(data.shape[0]):
        data[i] -= X_norm.reshape((data.shape[1:]),order='F')
    return data, X_norm


def rmsep(y_true, y_pred):
    """Compute Root Mean Square Error between two arrays."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def rmse(y_true, y_pred):
    """Compute Root Mean Square Percentage Error between two arrays."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def qsquared(y_true, y_pred):
    """Compute the Q^2 Error between two arrays."""
    return 1 - ((torch.norm(y_true - y_pred) ** 2) / (torch.norm(y_true) ** 2))


def cov(A, B):
    """Computes the mode 1 (mode 0 in python) contraction of 2 matrices."""
    assert A.shape[0] == B.shape[0], "A and B need to have the same shape on axis 0"
    dimension_A = A.shape[1:]
    dimension_B = B.shape[1:]
    dimensions = list(dimension_A) + list(dimension_B)
    rmode_A = len(dimension_A)
    dim = A.shape[0]
    C = tl.zeros(dimensions)
    indices = []
    for mode in dimensions:
        indices.append(range(mode))
    for idx in product(*indices):
        idx_A, idx_B = list(idx[:rmode_A]), list(idx[rmode_A:])
        C[idx] = np.sum(
            [A[tuple([i] + idx_A)] * B[tuple([i] + idx_B)] for i in range(dim)]
        )
    return C


class HOPLS:
    def __init__(self, R, Ln, Km=None, metric=None, epsilon=0):
        """
        Parameters:
            R: int, The number of latent vectors.

            Ln: list, the ranks for the decomposition of X: [L2, ..., LN].

            Km: list, the ranks for the decomposition of Y: [K2, ..., KM].

            epsilon: Float, default: 10e-7, The implicit secondary criterion of the
                algorithm. The algorithm will stop if we have not reached R but the
                residuals have a norm smaller than epsilon.
        """
        self.R = R
        self.Ln = Ln
        self.N = len(self.Ln)
        self.Km = Km
        if Km is not None:
            self.M = len(self.Km)
        else:
            self.M = 2
        self.epsilon = epsilon
        if metric is None:
            self.metric = qsquared
        else:
            self.metric = metric
        self.model = None
        self.mY = None

    def _fit_2d(self, X, Y):
        """
        Compute the HOPLS for X and Y wrt the parameters R, Ln and Km for the special case mode_Y = 2.

        Parameters:
            X: tensorly Tensor, The target tensor of shape [i1, ... iN], N = 2.

            Y: tensorly Tensor, The target tensor of shape [j1, ... jM], M >= 3.

        Returns:
            G: Tensor, The core Tensor of the HOPLS for X, of shape (R, L2, ..., LN).

            P: List, The N-1 loadings of X.

            D: Tensor, The core Tensor of the HOPLS for Y, of shape (R, K2, ..., KN).

            Q: List, The N-1 loadings of Y.

            ts: Tensor, The latent vectors of the HOPLS, of shape (i1, R).
        """

        # Initialization
        X, _ = remove_mean(X)
        Y, self.mY = remove_mean(Y)
        In = X.shape[1:]
        I0 = X.shape[0]
        Er, Fr = X, Y
        P, T = [], []
        Q = tl.zeros((Y.shape[-1], self.R))
        G = tl.zeros((self.R, *self.Ln))
        D = tl.zeros((self.R, self.R))
        W = tl.zeros((*In, self.R)).reshape(-1, self.R)
        Gr, _ = tucker(Er, ranks=[1] + self.Ln)

        # Beginning of the algorithm
        for r in range(self.R):
            if torch.norm(Er) > self.epsilon and torch.norm(Fr) > self.epsilon:
                # computing the covariance
                Cr = mode_dot(Er, Fr.transpose(0, 1), 0)

                # HOOI tucker decomposition of C
                _, latents = tucker(Cr, ranks=[1] + self.Ln)

                # Getting P and Q loadings
                qr = latents[0]
                Pr = latents[1:]
                tr = multi_mode_dot(Er, Pr, list(range(1, len(Pr) + 1)), transpose=True)
                Gr_pi = torch.pinverse(matricize(Gr))
                tr = torch.mm(matricize(tr), Gr_pi)
                tr /= torch.norm(tr)

                # recomposition of the core tensor of Y
                ur = torch.mm(Fr, qr)
                dr = torch.mm(ur.transpose(0, 1), tr)
                D[r, r] = dr

                Pkron = kronecker([Pr[self.N - n - 1] for n in range(self.N)])
                W[:, r] = torch.mm(Pkron, Gr_pi).view(1, -1)
                Pr = torch.mm(matricize(Gr), Pkron.transpose(0, 1))

                # Gathering of R variables
                Q[:, r] = qr.view(-1)
                D[r, r] = dr
                T.append(tr)
                temp_T = torch.cat(T, dim=1)
                P.append(Pr)
                temp_P = torch.cat(P)

                # Deflation
                X_hat = torch.mm(temp_T, temp_P)
                Er = X - np.reshape(X_hat, (Er.shape), order="F")
                Fr = Fr - dr * torch.mm(tr, qr.transpose(0, 1))
            else:
                break
        self.model = (temp_P, Q, G, D, temp_T, W)
        return self

    def fit(self, X, Y):
        """
        Compute the HOPLS for X and Y wrt the parameters R, Ln and Km.

        Parameters:
            X: tensorly Tensor, The target tensor of shape [i1, ... iN], N >= 3.

            Y: tensorly Tensor, The target tensor of shape [j1, ... jM], M >= 3.

        Returns:
            G: Tensor, The core Tensor of the HOPLS for X, of shape (R, L2, ..., LN).

            P: List, The N-1 loadings of X.

            D: Tensor, The core Tensor of the HOPLS for Y, of shape (R, K2, ..., KN).

            Q: List, The N-1 loadings of Y.

            ts: Tensor, The latent vectors of the HOPLS, of shape (i1, R).
        """
        # check parameters
        X_mode = len(X.shape)
        Y_mode = len(Y.shape)
        assert Y_mode >= 2, "Y need to be mode 2 minimum."
        assert X_mode >= 3, "X need to be mode 3 minimum."
        assert (
            len(self.Ln) == X_mode - 1
        ), f"The list of ranks for the decomposition of X (Ln) need to be equal to the mode of X -1: {X_mode-1}."
        if Y_mode == 2:
            return self._fit_2d(X, Y)
        assert (
            len(self.Km) == Y_mode - 1
        ), f"The list of ranks for the decomposition of Y (Km) need to be equal to the mode of Y -1: {Y_mode-1}."

        # Initialization
        Er, Fr = X, Y
        In = X.shape
        P, Q = [], []
        G = tl.zeros((self.R, *self.Ln))
        D = tl.zeros((self.R, *self.Km))
        T = tl.zeros((In[0], self.R))

        # Beginning of the algorithm
        for r in range(self.R):
            if torch.norm(Er) > self.epsilon and torch.norm(Fr) > self.epsilon:
                Cr = cov(Er, Fr)
                # HOOI tucker decomposition of C
                _, latents = tucker(
                    Cr, ranks=self.Ln + self.Km, n_iter_max=int(1e6), tol=1e-7
                )

                # Getting P and Q loadings
                Pr = latents[: len(Er.shape) - 1]
                Qr = latents[len(Er.shape) - 1 :]

                # computing product of Er by latents of X
                tr = multi_mode_dot(Er, Pr, list(range(1, len(Pr))), transpose=True)

                # Getting t as the first leading left singular vector of the product
                tr = torch.svd(tl.unfold(tr, 0))[0][:, 0]
                tr = tr[..., np.newaxis]

                # recomposition of the core tensors
                Gr = tl.tucker_to_tensor(Er, [tr] + Pr, transpose_factors=True)
                Dr = tl.tucker_to_tensor(Fr, [tr] + Qr, transpose_factors=True)

                # Gathering of
                P.append(Pr)
                Q.append(Qr)
                G[r] = Gr[0]
                D[r] = Dr[0]
                T[:, r] = tr[:, 0]

                # Deflation
                Er = Er - tl.tucker_to_tensor(Gr, [tr] + Pr)
                Fr = Fr - tl.tucker_to_tensor(Dr, [tr] + Qr)
            else:
                break
        self.model = (P, Q, G, D, T)
        return self

    def predict(self, X, Y):
        """Compute the HOPLS for X and Y wrt the parameters R, Ln and Km.

        Parameters:
            X: tensorly Tensor, The tensor we wish to do a prediction from.
            Of shape [i1, ... iN], N >= 3.

            Y: tensorly Tensor, used only for the shape of the prediction.

        Returns:
            Y_pred: tensorly Tensor, The predicted Y from the model.
        """
        P, Q, G, D, T, W = self.model
        Wfin = []
        for r in range(self.R):
            inter = torch.pinverse(
                torch.mm(P[: r + 1, :], W[:, : r + 1])
                + 2e-16 * torch.diag(torch.rand(r + 1))
            )
            W_star = torch.mm(W[:, : r + 1], inter)
            Wfin.append(
                torch.mm(
                    W_star, torch.mm(D[: r + 1, : r + 1], Q[:, : r + 1].transpose(0, 1))
                )
            )
        best_q2 = 0
        for r in range(self.R):
            Y_pred = torch.mm(matricize(X), Wfin[r])
            Y_pred, _ = remove_mean(Y_pred, self.mY)
            Q2 = qsquared(Y, Y_pred)
            if Q2 > best_q2:
                best_q2 = Q2
                best_r = r + 1
                best_Y_pred = Y_pred

        return best_Y_pred, best_r, best_q2

        # G_pi = []
        # if len(Y.shape) == 2:
        #     Q_star = tl.zeros(Q.shape)
        # else:
        #     Q_star = []
        # W = tl.zeros((*X.shape[1:], self.R)).reshape(-1, self.R)
        # for r in range(self.R):
        #     G_pi.append(pinverse(G[r]))
        #     W[:, r] = torch.mm(
        #         kronecker([P[r][self.N - n - 1] for n in range(self.N)]),
        #         G_pi[-1].reshape(-1),
        #     )
        #     if len(Y.shape) > 2:
        #         Q_star.append(
        #             torch.mm(
        #                 D[r].reshape(-1),
        #                 kronecker([Q[r][self.M - n - 1] for n in range(self.M)]).T,
        #             )
        #         )
        #     else:
        #         Q_star[:, r] = D[r, r] * Q[:, r]

        # if len(Y.shape) > 2:
        #     Q_star = tl.tensor(Q_star).T
        # return torch.mm(torch.mm(tl.unfold(X, 0), W), Q_star.T).reshape(Y.shape)
        # return torch.mm(T, Q_star.T).reshape(Y.shape)

    def score(self, X, Y):
        self.fit(X, Y)
        Y_pred = self.predict(X, Y)
        return self.metric(Y.reshape(Y.shape[0], -1), Y_pred.reshape(Y.shape[0], -1))
