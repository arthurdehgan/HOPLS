from itertools import product
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg.n_mode_product import mode_dot
from tensorly.tenalg import kronecker
import numpy as np
from numpy.linalg import svd, pinv


def rmsep(y_true, y_pred):
    """Compute Root Mean Square Percentage Error between two arrays."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def qsquared(y_true, y_pred):
    """Compute the Q^2 Error between two arrays."""
    return 1 - ((tl.norm(y_true - y_pred) ** 2) / (tl.norm(y_true) ** 2))


def cov(A, B):
    # """Computes the mode 1 (mode 0 in python) contraction of 2 matrices."""
    # assert A.shape[0] == B.shape[0], "A and B need to have the same shape on axis 0"
    # dimension_A = A.shape[1:]
    # dimension_B = B.shape[1:]
    # dimensions = list(dimension_A) + list(dimension_B)
    # rmode_A = len(dimension_A)
    # dim = A.shape[0]
    # C = np.zeros(dimensions)
    # indices = []
    # for mode in dimensions:
    #     indices.append(range(mode))
    # for idx in product(indices):
    #     idx_A, idx_B = list(idx[:rmode_A]), list(idx[rmode_A:])
    #     C[tuple(idx_A + idx_B)] = sum(
    #         [A[tuple([i] + idx_A)] * B[tuple([i] + idx_B)] for i in range(dim)]
    #     )
    # return C
    # alpha = "bcdefghijklmnopqrstuvwxyz"
    # A_alpha = alpha[: len(A.shape) - 1]
    # B_alpha = alpha[1 - len(B.shape) :]
    # string = "a" + A_alpha + "," + B_alpha + "->" + A_alpha + B_alpha
    # print(A.shape, B.shape)
    # return np.einsum("a...,a...->...", A, B)
    return


class HOPLS:
    def __init__(self, R, Ln, Kn=None, metric=None, epsilon=0):
        """
        Parameters:
            R: int, The number of latent vectors.

            Ln: list, the ranks for the decomposition of X: [L2, ..., LN].

            Kn: list, the ranks for the decomposition of Y: [K2, ..., KM].

            epsilon: Float, default: 10e-7, The implicit secondary criterion of the
                algorithm. The algorithm will stop if we have not reached R but the
                residuals have a norm smaller than epsilon.
        """
        self.R = R
        self.Ln = Ln
        self.Kn = Kn
        self.epsilon = epsilon
        if metric is None:
            self.metric = qsquared
        self.model = None

    def _fit_2d(self, X, Y):
        """
        Compute the HOPLS for X and Y wrt the parameters R, Ln and Kn for the special case mode_Y = 2.

        Parameters:
            X: tensorly Tensor, The target tensor of shape [i1, ... iN], N = 2.

            Y: tensorly Tensor, The target tensor of shape [j1, ... jM], M >= 3.

        Returns:
            G: Tensor, The core Tensor of the HOPLS for X, of shape (R, L2, ..., LN).

            P: List, The N-1 loadings of X. of shape (R, I(n+1), L(n+1)) for n from 1 to N-1.

            D: Tensor, The core Tensor of the HOPLS for Y, of shape (R, K2, ..., KN).

            Q: List, The N-1 loadings of Y.

            ts: Tensor, The latent vectors of the HOPLS, of shape (i1, R).
        """
        # Initialization
        In = X.shape
        M = Y.shape[-1]
        Er, Fr = X, Y
        P = []
        Q = tl.zeros((M, self.R))
        G = tl.zeros((self.R, *self.Ln))
        D = tl.zeros((self.R, self.R))
        T = tl.zeros((In[0], self.R))
        # Beginning of the algorithm
        for r in range(self.R):
            if tl.norm(Er) > self.epsilon and tl.norm(Fr) > self.epsilon:
                Cr = np.tensordot(Er, Fr, (0, 0))
                # HOOI tucker decomposition of C
                # print(len(self.Ln), len(Er.shape))
                Gr_C, latents = tucker(
                    Cr, ranks=[1] + self.Ln, n_iter_max=int(1e6), tol=1e-7
                )
                # Getting P and Q
                qr = latents[0]
                Pr = latents[1:]
                tr = Er
                for k in range(len(Er.shape) - 1):
                    tr = mode_dot(tr, Pr[k].T, k + 1)
                tr = np.matmul(tl.unfold(tr, 0), pinv(Gr_C.reshape(1, -1)))
                tr /= tl.norm(tr)
                # recomposition of the core tensors
                Gr = tl.tucker_to_tensor(Er, [tr.T] + [pn.T for pn in Pr])
                ur = np.matmul(Fr, qr)
                dr = np.matmul(ur.T, tr)
                # Deflation
                Er = Er - tl.tucker_to_tensor(Gr, [tr] + Pr)
                Fr = Fr - dr * np.matmul(tr, qr.T)
                # Gathering of R variables
                P.append(Pr)
                Q[:, r] = qr.T
                G[r] = Gr[0]
                D[r, r] = dr
                T[:, r] = tr[:, 0]
            else:
                break
        self.model = (P, Q, G, D, T)
        return self.model

    def fit(self, X, Y):
        """
        Compute the HOPLS for X and Y wrt the parameters R, Ln and Kn.

        Parameters:
            X: tensorly Tensor, The target tensor of shape [i1, ... iN], N >= 3.

            Y: tensorly Tensor, The target tensor of shape [j1, ... jM], M >= 3.

        Returns:
            G: Tensor, The core Tensor of the HOPLS for X, of shape (R, L2, ..., LN).

            P: List, The N-1 loadings of X. of shape (R, I(n+1), L(n+1)) for n from 1 to N-1.

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
            len(self.Kn) == Y_mode - 1
        ), f"The list of ranks for the decomposition of Y (Kn) need to be equal to the mode of Y -1: {Y_mode-1}."
        # Initialization
        Er, Fr = X, Y
        In = X.shape
        P, Q = [], []
        G = tl.zeros((self.R, *self.Ln))
        D = tl.zeros((self.R, *self.Kn))
        T = tl.zeros((In[0], self.R))
        # Beginning of the algorithm
        for r in range(self.R):
            if tl.norm(Er) > self.epsilon and tl.norm(Fr) > self.epsilon:
                Cr = np.tensordot(Er, Fr, (0, 0))
                # HOOI tucker decomposition of C
                _, latents = tucker(
                    Cr, ranks=self.Ln + self.Kn, n_iter_max=int(1e6), tol=1e-7
                )
                # Getting P and Q
                Pr = latents[: len(Er.shape) - 1]
                Qr = latents[len(Er.shape) - 1 :]
                # computing product of Er by latents of X
                tr = Er
                for k in range(len(Er.shape) - 1):
                    tr = mode_dot(tr, Pr[k].T, k + 1)
                # Getting t as the first leading left singular vector of the product
                tr = svd(tl.unfold(tr, 0))[0][:, 0]
                tr = tr[..., np.newaxis]
                # recomposition of the core tensors
                Gr = tl.tucker_to_tensor(Er, [tr.T] + [pn.T for pn in Pr])
                Dr = tl.tucker_to_tensor(Fr, [tr.T] + [qn.T for qn in Qr])
                # Deflation
                Er = Er - tl.tucker_to_tensor(Gr, [tr] + Pr)
                Fr = Fr - tl.tucker_to_tensor(Dr, [tr] + Qr)
                # Gathering of
                P.append(Pr)
                Q.append(Qr)
                G[r] = Gr[0]
                D[r] = Dr[0]
                T[:, r] = tr[:, 0]
            else:
                break
        self.model = (P, Q, G, D, T)
        return self.model

    def predict(self, X, Y):
        """Compute the HOPLS for X and Y wrt the parameters R, Ln and Kn.

        Parameters:
            X: tensorly Tensor, The tensor we wish to do a prediction from.
            Of shape [i1, ... iN], N >= 3.

            Y: tensorly Tensor, used only for the shape of the prediction.

        Returns:
            Y_pred: tensorly Tensor, The predicted Y from the model.
        """
        P, Q, G, D, _ = self.model
        N = len(self.Ln)
        G_pi = tl.zeros(G.shape)
        if len(Y.shape) == 2:
            Q_star = tl.zeros(Q.shape)
        else:
            Q_star = []
        W = tl.zeros((*X.shape[1:], self.R)).reshape(-1, self.R)
        for r in range(self.R):
            G_pi[r] = pinv(G[r])
            W[:, r] = np.matmul(
                kronecker([P[r][N - n - 1] for n in range(N)]), G_pi[r].reshape(-1)
            )
            if len(Y.shape) > 2:
                Q_star.append(
                    np.matmul(
                        D[r].reshape(-1),
                        kronecker([Q[r][N - n - 1] for n in range(N)]).T,
                    )
                )
            else:
                Q_star[:, r] = D[r, r] * Q[:, r]

        if len(Y.shape) > 2:
            Q_star = tl.tensor(Q_star).T
        print(tl.unfold(X, 0).shape, W.shape, Q_star.shape)
        return np.matmul(np.matmul(tl.unfold(X, 0), W), Q_star.T)

    def score(self, X, Y):
        self.fit(X, Y)
        Y_pred = self.predict(X, Y)
        return self.metric(Y.reshape(Y.shape[0], -1), Y_pred)
