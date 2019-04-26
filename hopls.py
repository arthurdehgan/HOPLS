"""
Implementation of HOPLS using torch and tntensor
https://github.com/rballester/tntorch

Author:
    Arthur Dehgan
"""
from itertools import product
import tntorch as tn
import torch
from torch import norm as frob_norm, mean, sqrt, svd, pinverse as pinv


def rmsep(y_true, y_pred):
    """Compute Root Mean Square Error between two arrays."""
    return sqrt(mean((y_true - y_pred) ** 2, dim=0))


def rmse(y_true, y_pred):
    """Compute Root Mean Square Percentage Error between two arrays."""
    return sqrt(mean((y_true - y_pred) ** 2, dim=0))


def qsquared(y_true, y_pred):
    """Compute the Q^2 Error between two arrays."""
    return 1 - ((frob_norm(y_true - y_pred) ** 2) / (frob_norm(y_true) ** 2))


def prod(lis):
    ret = 1
    for i in lis:
        ret *= i
    return ret


def mode_product(A, B, mode=0):
    A = torch.tensor(A)
    B = torch.tensor(B)
    dimA = A.shape[:mode] + A.shape[mode + 1 :]
    dimB = B.shape[:mode] + B.shape[mode + 1 :]
    return torch.mm(
        A.resize_(prod(dimA), A.shape[mode]), B.resize_(B.shape[mode], prod(dimB))
    ).view(dimA, dimB)


def tensor_dot(A, B, mode=0):
    """Computes the mode 1 (mode 0 in python) contraction of 2 matrices."""
    assert A.shape[0] == B.shape[0], "A and B need to have the same shape on axis 0"
    dimA = A.shape[:mode] + A.shape[mode + 1 :]
    dimB = B.shape[:mode] + B.shape[mode + 1 :]
    dimensions = list(dimA) + list(dimB)
    rmode_A = len(dimA)
    dim = A.shape[mode]
    C = torch.zeros(dimensions)
    indices = []
    for mode in dimensions:
        indices.append(range(mode))
    for idx in product(*indices):
        idx_A, idx_B = list(idx[:rmode_A]), list(idx[rmode_A:])
        som = torch.Tensor(
            [A[tuple([i] + idx_A)] * B[tuple([i] + idx_B)] for i in range(dim)]
        )
        C[idx] = torch.sum(som)
    return C


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
        else:
            self.metric = metric
        self.model = None

    def _fit_2d(self, X, Y):
        """
        Compute the HOPLS for X and Y wrt the parameters R, Ln and Kn for the special case mode_Y = 2.

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
        In = X.shape
        M = Y.shape[-1]
        Er, Fr = X, Y
        P = []
        Q = torch.zeros((M, self.R))
        G = torch.zeros((self.R, *self.Ln))
        D = torch.zeros((self.R, self.R))
        T = torch.zeros((In[0], self.R))

        # Beginning of the algorithm
        for r in range(self.R):
            if frob_norm(Er) > self.epsilon and frob_norm(Fr) > self.epsilon:
                # computing the covariance
                Cr = cov(Er, Fr)

                # HOOI tucker decomposition of C
                decomp = tn.Tensor(Cr, ranks_tucker=[1] + self.Ln)

                # Getting P and Q loadings
                qr = decomp.Us[0]
                Pr = decomp.Us[1:]
                tr = Er
                for i, p in enumerate(Pr):
                    tr = tn.dot(tr, p, k=i + 2)
                tr = multi_mode_dot(Er, Pr, list(range(1, len(Pr))), transpose=True)
                tr = np.matmul(tl.unfold(tr, 0), pinv(Gr_C.reshape(1, -1)))
                tr /= frob_norm(tr)

                # recomposition of the core tensors
                Gr = tl.tucker_to_tensor(Er, [tr] + Pr, transpose_factors=True)
                ur = np.matmul(Fr, qr)
                dr = np.matmul(ur.T, tr)

                # Gathering of R variables
                P.append(Pr)
                Q[:, r] = qr.T
                G[r] = Gr[0]
                D[r, r] = dr
                T[:, r] = tr[:, 0]

                # Deflation
                Er = Er - tl.tucker_to_tensor(Gr, [tr] + Pr)
                Fr = Fr - dr * np.matmul(tr, qr.T)
            else:
                break
        self.model = (P, Q, G, D, T)
        return self

    def fit(self, X, Y):
        """
        Compute the HOPLS for X and Y wrt the parameters R, Ln and Kn.

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
            len(self.Kn) == Y_mode - 1
        ), f"The list of ranks for the decomposition of Y (Kn) need to be equal to the mode of Y -1: {Y_mode-1}."

        # Initialization
        Er, Fr = X, Y
        In = X.shape
        P, Q = [], []
        G = torch.zeros((self.R, *self.Ln))
        D = torch.zeros((self.R, *self.Kn))
        T = torch.zeros((In[0], self.R))

        # Beginning of the algorithm
        for r in range(self.R):
            if frob_norm(Er) > self.epsilon and frob_norm(Fr) > self.epsilon:
                Cr = cov(Er, Fr)
                # HOOI tucker decomposition of C
                _, latents = tucker(
                    Cr, ranks=self.Ln + self.Kn, n_iter_max=int(1e6), tol=1e-7
                )

                # Getting P and Q loadings
                Pr = latents[: len(Er.shape) - 1]
                Qr = latents[len(Er.shape) - 1 :]

                # computing product of Er by latents of X
                tr = multi_mode_dot(Er, Pr, list(range(1, len(Pr))), transpose=True)

                # Getting t as the first leading left singular vector of the product
                tr = svd(tl.unfold(tr, 0))[0][:, 0]
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
        """Compute the HOPLS for X and Y wrt the parameters R, Ln and Kn.

        Parameters:
            X: tensorly Tensor, The tensor we wish to do a prediction from.
            Of shape [i1, ... iN], N >= 3.

            Y: tensorly Tensor, used only for the shape of the prediction.

        Returns:
            Y_pred: tensorly Tensor, The predicted Y from the model.
        """
        P, Q, G, D, T = self.model
        N = len(self.Ln)
        M = len(self.Kn)
        G_pi = []
        if len(Y.shape) == 2:
            Q_star = torch.zeros(Q.shape)
        else:
            Q_star = []
        W = torch.zeros((*X.shape[1:], self.R)).reshape(-1, self.R)
        for r in range(self.R):
            G_pi.append(pinv(G[r]))
            W[:, r] = np.matmul(
                kronecker([P[r][N - n - 1] for n in range(N)]), G_pi[-1].reshape(-1)
            )
            if len(Y.shape) > 2:
                Q_star.append(
                    np.matmul(
                        D[r].reshape(-1),
                        kronecker([Q[r][M - n - 1] for n in range(M)]).T,
                    )
                )
            else:
                Q_star[:, r] = D[r, r] * Q[:, r]

        if len(Y.shape) > 2:
            Q_star = tl.tensor(Q_star).T
        return np.matmul(np.matmul(tl.unfold(X, 0), W), Q_star.T).reshape(Y.shape)
        # return np.matmul(T, Q_star.T).reshape(Y.shape)

    def score(self, X, Y):
        self.fit(X, Y)
        Y_pred = self.predict(X, Y)
        return self.metric(Y.reshape(Y.shape[0], -1), Y_pred.reshape(Y.shape[0], -1))
