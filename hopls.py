from itertools import product
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg.n_mode_product import mode_dot
from tensorly.tenalg import kronecker
import numpy as np
from numpy.linalg import svd, pinv


def cov(A, B):
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
        X: tensorly Tensor, The data tensor, needs to be of order 3 or more.

        Y: tensorly Tensor, The target tensor, needs to be of order 3 or more.

        R: int, The number of latent vectors.

        Ln: list, the ranks for the decomposition of X.

        Kn: list, the ranks for the decomposition of Y.

        epsilon: Float, default: 10e-5, The implicit secondary criterion of the
            algorithm. The algorithm will stop if we have not reached R but the
            residuals have a norm smaller than epsilon.

    Returns:
        G: The core Tensor of the HOPLS for X.

        P: The loadings of X.

        D: The core Tensor of Y.

        Q: The loadings of Y.

        ts: The latent vectors of the HOPLS.
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
    E, F = X, Y
    Pr, Qr, Gr, Dr = [], [], [], []
    ts = list()
    for i in range(R):
        if tl.norm(E) > epsilon and tl.norm(F) > epsilon:
            C = cov(E, F)
            _, latents = tucker(C, ranks=Ln + Kn)
            P = latents[: X_mode - 1]
            Q = latents[X_mode - 1 :]
            t = svd(E)[0]
            while len(t.shape) > 1:
                t = t[:, 0]
            t = t[..., np.newaxis]
            Gr.append(tl.tucker_to_tensor(E, [t.T] + [p.T for p in P]))
            Dr.append(tl.tucker_to_tensor(F, [t.T] + [q.T for q in Q]))
            E = E - tl.tucker_to_tensor(Gr[i], [t] + P)
            F = F - tl.tucker_to_tensor(Dr[i], [t] + Q)
            Pr.append(P)
            Qr.append(Q)
            ts.append(t)
        else:
            R = i
            break
    return (
        tl.tensor(Gr).squeeze(),
        tl.tensor(Pr),
        tl.tensor(Dr).squeeze(),
        tl.tensor(Qr),
        tl.tensor(ts).squeeze(),
        R,
    )


if __name__ == "__main__":
    epsilon = 1e-5
    T = tl.tensor(np.random.normal(size=(20, 5)))
    P = tl.tensor(np.random.normal(size=(5, 10, 10)))
    Q = tl.tensor(np.random.normal(size=(5, 10, 10)))
    E = tl.tensor(np.random.normal(size=(20, 10, 10)))
    F = tl.tensor(np.random.normal(size=(20, 10, 10)))
    X = mode_dot(P, T, 0) + epsilon * E
    Y = mode_dot(Q, T, 0) + epsilon * F
    old_mse = np.inf

    for R, l in product(range(2, 10), range(2, 10)):
        g, p, d, q, ts, R = hopls(X, Y, R, [l, l], [l, l])
        gp, wr, qr = [], [], []
        Y_pred = tl.tensor(np.zeros(Y.shape))
        if R > 1:
            for k in range(R):
                comp = mode_dot(d[k][np.newaxis, ...], ts[k][..., np.newaxis], 0)
                for j in range(q.shape[1]):
                    comp = mode_dot(comp, q[k, j], j + 1)
                Y_pred += comp
        else:
            g = g[np.newaxis, ...]
            d = d[np.newaxis, ...]
            ts = ts[..., np.newaxis]
            comp = mode_dot(d, ts, 0)
            for j in range(q.shape[1]):
                comp = mode_dot(comp, q[:, j].squeeze(), j + 1)
            Y_pred += comp
            # gp.append(pinv(g))
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
