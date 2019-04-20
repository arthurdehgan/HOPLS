from itertools import product
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg.n_mode_product import mode_dot
from tensorly.tenalg import kronecker
import numpy as np
from numpy.linalg import svd, pinv
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression


def rmse(y_true, y_pred):
    """Compute Root Mean Square Percentage Error between two arrays."""
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_pred)), axis=0))


def cov(A, B):
    """Computes the mode 1 (mode 0 in python) contraction of 2 matrices."""
    assert A.shape[0] == B.shape[0], "A and B need to have the same shape on axis 0"
    dimension_A = A.shape[1:]
    dimension_B = B.shape[1:]
    dimensions = list(dimension_A) + list(dimension_B)
    rmode_A = len(dimension_A)
    dim = A.shape[0]
    C = np.zeros(dimensions)
    indices = []
    for mode in dimensions:
        indices.append(range(mode))
    for idx in product(indices):
        idx_A, idx_B = list(idx[:rmode_A]), list(idx[rmode_A:])
        C[tuple(idx_A + idx_B)] = sum(
            [A[tuple([i] + idx_A)] * B[tuple([i] + idx_B)] for i in range(dim)]
        )
    return C


class HOPLS:
    def __init__(self, R, epsilon=1e-15):
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
        self.epsilon = epsilon

    def _fit_2d(self, X, Y, Ln):
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
        Er, Fr = X, Y
        P, Q, G, D, T = [], [], [], [], []
        # Beginning of the algorithm
        for i in range(self.R):
            if tl.norm(Er) > epsilon and tl.norm(Fr) > epsilon:
                Cr = cov(Er, Fr)
                # HOOI ticker decomposition of C
                Gr_C, latents = tucker(Cr, ranks=[1] + Ln)
                # Getting P and Q
                qr = latents[0]
                Pr = latents[1:]
                tr = Er
                for k in range(len(Er.shape) - 1):
                    tr = mode_dot(tr, Pr[k].T, k + 1)
                tr = np.matmul(tl.unfold(tr, 0), pinv(Gr_C.reshape(1, -1)))
                tr /= tl.norm(tr)
                # recomposition of the core tensors
                G.append(tl.tucker_to_tensor(Er, [tr.T] + [pn.T for pn in Pr]))
                ur = np.matmul(Fr, qr)
                dr = np.matmul(ur.T, tr)
                # Deflation
                Er = Er - tl.tucker_to_tensor(G[i], [tr] + Pr)
                Fr = Fr - dr * np.matmul(tr, qr.T)
                # Gathering of R variables
                P.append(Pr)
                Q.append(qr)
                D.append(dr)
                T.append(tr)
            else:
                break
        # reshaping the loadings
        P = tl.tensor(P)
        Q = tl.tensor(Q)
        # P = [P[:, i] for i in range(P.shape[1])]
        # Q = [Q[:, i] for i in range(Q.shape[1])]
        self.model = (
            tl.tensor(G).squeeze(),
            P,
            tl.tensor(D).squeeze(),
            Q,
            tl.tensor(T).squeeze(),
        )
        return self.model

    def fit(self, X, Y, Ln, Kn=None):
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
            len(Ln) == X_mode - 1
        ), f"The list of ranks for the decomposition of X (Ln) need to be equal to the mode of X -1: {X_mode-1}."
        if Y_mode == 2:
            return self._fit_2d(X, Y, Ln)
        assert (
            len(Kn) == Y_mode - 1
        ), f"The list of ranks for the decomposition of Y (Ln) need to be equal to the mode of Y -1: {Y_mode-1}."
        # Initialization
        Er, Fr = X, Y
        P, Q, G, D, T = [], [], [], [], []
        # Beginning of the algorithm
        for i in range(self.R):
            if tl.norm(Er) > epsilon and tl.norm(Fr) > epsilon:
                Cr = cov(Er, Fr)
                # HOOI ticker decomposition of C
                _, latents = tucker(Cr, ranks=Ln + Kn)
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
                G.append(tl.tucker_to_tensor(Er, [tr.T] + [pn.T for pn in Pr]))
                D.append(tl.tucker_to_tensor(Fr, [tr.T] + [qn.T for qn in Qr]))
                # Deflation
                Er = Er - tl.tucker_to_tensor(G[i], [tr] + Pr)
                Fr = Fr - tl.tucker_to_tensor(D[i], [tr] + Qr)
                # Gathering of
                P.append(Pr)
                Q.append(Qr)
                T.append(tr)
            else:
                break
        # reshaping the loadings
        P = tl.tensor(P)
        Q = tl.tensor(Q)
        # P = [P[:, i] for i in range(P.shape[1])]
        # Q = [Q[:, i] for i in range(Q.shape[1])]
        self.model = (
            tl.tensor(G).squeeze(),
            P,
            tl.tensor(D).squeeze(),
            Q,
            tl.tensor(T).squeeze(),
        )
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
        g, p, d, q, ts = self.model
        N = p.shape[1]
        gp = [pinv(g[k]) for k in range(self.R)]
        wr = [
            np.matmul(kronecker([p[k, N - i - 1] for i in range(N)]), gp[k].flatten())
            for k in range(self.R)
        ]
        qr = [
            np.matmul(d[k].flatten(), kronecker([q[k, N - i - 1] for i in range(N)]).T)
            for k in range(self.R)
        ]
        W_star = np.asarray(wr).T
        Q_star = np.asarray(qr).T
        X = tl.unfold(X, 0)
        X = np.matmul(X, W_star)
        if len(Y.shape) > 2:
            return np.matmul(X, Q_star.T).reshape(Y.shape)
        elif len(Y.shape) == 2:
            return np.matmul(np.matmul(X, np.diag(d)), q.squeeze())

    def score(self, X, Y, Ln, Kn=None, metric=None):
        self.fit(X, Y, Ln, Kn)
        Y_pred = self.predict(X, Y)
        if metric is None:
            metric = rmse
        return metric(Y, Y_pred)


if __name__ == "__main__":
    # arbitrarly chosen epsilon
    epsilon = 5e-4
    # Generation according to 4.1.1 of the paper equation (29)
    T = tl.tensor(np.random.normal(size=(20, 5)))
    P = tl.tensor(np.random.normal(size=(5, 10, 10)))
    Q = tl.tensor(np.random.normal(size=(5, 10)))
    E = tl.tensor(np.random.normal(size=(20, 10, 10)))
    F = tl.tensor(np.random.normal(size=(20, 10, 10)))
    X = mode_dot(P, T, 0)  # + epsilon * E
    Y = mode_dot(Q, T, 0)  # + epsilon * F
    old_mse = np.inf

    # Just like in the paper, we simplify by having l for all ranks
    for R, l in product(range(3, 7), range(3, 10)):
        hopls = HOPLS(R)
        rmsep = np.mean(hopls.score(X, Y, [l, l], [l, l]))

        if rmsep < old_mse:
            old_mse = rmsep
            best_params = [R, l, rmsep]

    print("Best model is with R={} and l={}, rmsep={:.2f}".format(*best_params))
