import numpy as np
import tensorly as tl
from tensorly.tenalg.n_mode_product import mode_dot
from scipy.io import savemat


if __name__ == "__main__":
    # Generation according to 4.1.1 of the paper equation (29)
    T = tl.tensor(np.random.normal(size=(100, 5)))
    P = tl.tensor(np.random.normal(size=(5, 10, 10, 15, 10)))
    Q = tl.tensor(np.random.normal(size=(5, 20)))
    E = tl.tensor(np.random.normal(size=(100, 10, 10, 15, 10)))
    F = tl.tensor(np.random.normal(size=(100, 20)))
    X = mode_dot(P, T, 0)
    Y = mode_dot(Q, T, 0)

    for i, snr in enumerate([10, 5, 0, -5, -10]):
        epsilon = 1 / (10 ** (snr / 10))
        noisy_X = X + epsilon * E
        noisy_Y = Y + epsilon * F
        savemat(
            f"hox_data_{snr}dB",
            {
                "data": noisy_X,
                "target": noisy_Y,
                "latents": T,
                "Q": Q,
                "P": P,
                "E": E,
                "F": F,
            },
        )
