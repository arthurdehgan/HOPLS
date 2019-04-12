from itertools import product
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg.n_mode_product import mode_dot
from tensorly.tenalg import kronecker
import numpy as np
from numpy.linalg import svd, pinv


def cov(A, B):
    "Computes the mode 1 (mode 0 in python) contraction of 2 matrices."
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


def hopls(X, Y, R, Ln, Kn, epsilon=10e-7):
    """Compute the HOPLS for X and Y wrt the parameters R, Ln and Kn.

    Parameters:
        X: tensorly Tensor, The target tensor of shape [i1, ... iN], N >= 3.

        Y: tensorly Tensor, The target tensor of shape [j1, ... jM], M >= 3.

        R: int, The number of latent vectors.

        Ln: list, the ranks for the decomposition of X: [L2, ..., LN].

        Kn: list, the ranks for the decomposition of Y: [K2, ..., KM].

        epsilon: Float, default: 10e-7, The implicit secondary criterion of the
            algorithm. The algorithm will stop if we have not reached R but the
            residuals have a norm smaller than epsilon.

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
    assert Y_mode >= 3 and X_mode >= 3, "X and Y need to be mode 3 minimum."
    assert (
        len(Ln) == X_mode - 1
    ), f"The ranks for the decomposition of X (Ln) need to be of len {X_mode-1}."
    assert (
        len(Kn) == Y_mode - 1
    ), f"The ranks for the decomposition of Y (Ln) need to be of len {Y_mode-1}."
    # Initialization
    Er, Fr = X, Y
    P, Q, G, D, T = [], [], [], [], []
    # Beginning of the algorithm
    for i in range(R):
        if tl.norm(Er) > epsilon and tl.norm(Fr) > epsilon:
            Cr = cov(Er, Fr)
            # HOOI ticker decomposition of C
            _, latents = tucker(Cr, ranks=Ln + Kn)
            # Getting P and Q
            Pr = latents[: X_mode - 1]
            Qr = latents[X_mode - 1 :]
            # Getting t as the first leading left singular vector of E
            tr = svd(Er)[0]
            while len(tr.shape) > 1:
                tr = tr[:, 0]
            tr = tr[..., np.newaxis]
            # recomposition of the core tensors
            G.append(tl.tucker_to_tensor(E, [tr.T] + [pn.T for pn in Pr]))
            D.append(tl.tucker_to_tensor(F, [tr.T] + [qn.T for qn in Qr]))
            # Deflation
            Er = Er - tl.tucker_to_tensor(G[i], [tr] + Pr)
            Fr = Fr - tl.tucker_to_tensor(D[i], [tr] + Qr)
            # Gathering of
            P.append(Pr)
            Q.append(Qr)
            T.append(tr)
        else:
            R = i
            break
    # reshaping the loadings
    P = tl.tensor(P)
    Q = tl.tensor(Q)
    P = [P[:, i] for i in range(P.shape[1])]
    Q = [Q[:, i] for i in range(Q.shape[1])]
    return (
        tl.tensor(G).squeeze(),
        P,
        tl.tensor(D).squeeze(),
        Q,
        tl.tensor(T).squeeze(),
        R,
    )


if __name__ == "__main__":
    # arbitrarly chosen epsilon
    epsilon = 1e-5
    # Generation according to 4.1.1 of the paper equation (29)
    T = tl.tensor(np.random.normal(size=(20, 5)))
    P = tl.tensor(np.random.normal(size=(5, 10, 10)))
    Q = tl.tensor(np.random.normal(size=(5, 10, 10)))
    E = tl.tensor(np.random.normal(size=(20, 10, 10)))
    F = tl.tensor(np.random.normal(size=(20, 10, 10)))
    X = mode_dot(P, T, 0) + epsilon * E
    Y = mode_dot(Q, T, 0) + epsilon * F
    old_mse = np.inf

    # Just like in the paper, we simplify by having l for all ranks
    for R, l in product(range(2, 10), range(2, 10)):
        g, p, d, q, ts, R = hopls(X, Y, R, [l, l], [l, l])
        Y_pred = tl.tensor(np.zeros(Y.shape))

        # we split R in two cases because if R = 1 the tensors being squeezed, there is no first
        # dimension
        if R > 1:
            # Recomposition of Y using equation (12) in section 3.1
            # Recomposing using (26), (27) in section 3.4 didn't work because of different shapes
            # and lack of information.
            for k in range(R):
                comp = mode_dot(d[k][np.newaxis, ...], ts[k][..., np.newaxis], 0)
                for j, qq in enumerate(q):
                    comp = mode_dot(comp, qq[k], j + 1)
                Y_pred += comp

        # case R = 1
        else:
            g = g[np.newaxis, ...]
            d = d[np.newaxis, ...]
            ts = ts[..., np.newaxis]
            comp = mode_dot(d, ts, 0)
            for j, qq in enumerate(q):
                comp = mode_dot(comp, qq.squeeze(), j + 1)
            Y_pred += comp

        # Evaluating performances using RMSEP
        mse = np.mean((np.square(Y - Y_pred)).mean(axis=0))
        if mse < old_mse:
            old_mse = mse
            best_params = [R, l, mse]
    print("Best model is with R={} and l={}, mse={:.2f}".format(*best_params))

    # wr.append(np.matmul(kronecker(p[k, i] for i in range(p.shape[1])), gp))
    # qr.append(np.matmul(d[k], kronecker([q[k, i] for i in range(q.shape[1])]).T))
    # W = np.asarray(wr)
    # Q_star = np.asarray(qr)
    # Y_pred = np.matmul(Q_star.T, ts)
