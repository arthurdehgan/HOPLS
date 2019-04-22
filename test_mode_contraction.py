"""
We are only interrested in mode 1 contraction (so mode 0 in numy since indexes start at 0)

We will compare the correct way (done with a for loop, according to the paper)
to the following functions:

mode_dot(A, B, 0) (Only works if B is 2D)
np.einsum
np.tensordot(A, B, (0,0))

"""

from itertools import product
import numpy as np
import tensorly as tl
from tensorly.tenalg.n_mode_product import mode_dot
from hopls import cov

print(__doc__)

In = [20, 10, 11]
Jn = [20, 12, 13]
N_latents = 5
T = tl.tensor(np.random.normal(size=(In[0], N_latents)))
P = tl.tensor(np.random.normal(size=(N_latents, In[1], In[2])))
Q_2D = tl.tensor(np.random.normal(size=(N_latents, Jn[1])))
Q = tl.tensor(np.random.normal(size=(N_latents, Jn[1], Jn[2])))

X = mode_dot(P, T, 0)
X = np.ones(In)
print(f"shape of the X tensor: {X.shape}")
Y = mode_dot(Q, T, 0)
print(f"shape of the Y tensor: {Y.shape}")
Y_2D = mode_dot(Q_2D, T, 0)
Y_2D = np.multiply(np.ones((12, 20)), np.arange(20)).T
print(f"shape of the Y tensor in the 2D case: {Y_2D.shape}")

# The correct way:
contraction_shape = (In[1], In[2], Jn[1], Jn[2])
print(contraction_shape)
test_forloop = tl.zeros(contraction_shape)
for i, j, k, l in product(range(In[1]), range(In[2]), range(Jn[1]), range(Jn[2])):
    test_forloop[i, j, k, l] = np.sum(
        [X[m, i, j] * Y[m, k, l] for m in range(X.shape[0])]
    )

# The correct way for the 2d case
contraction_shape_2D = (In[1], In[2], Jn[1])
print(contraction_shape_2D)
test_forloop_2D = tl.zeros(contraction_shape_2D)
for i, j, k in product(range(In[1]), range(In[2]), range(Jn[1])):
    test_forloop_2D[i, j, k] = np.sum([X[m, i, j] * Y[m, k] for m in range(X.shape[0])])

# with mode_dot, for the 2D case:
test_modedot = mode_dot(X, Y_2D.T, 0)
test_modedot = np.moveaxis(test_modedot, 0, -1)
print(test_modedot.shape)
if np.alltrue(test_modedot == test_forloop_2D):
    print("mode_dot is correct!")
else:
    print("mode_dot is incorrect")

# with np.einsum:
test_einsum = np.einsum("abc,ade->bcde", X, Y)
print(test_einsum.shape)
if np.alltrue(test_einsum == test_forloop):
    print("einsum is correct!")
else:
    print("einsum is incorrect")

# with np.einsum in 2D:
test_einsum_2D = np.einsum("abc,ad->bcd", X, Y_2D)
print(test_einsum_2D.shape)
if np.alltrue(test_einsum_2D == test_forloop_2D):
    print("einsum in 2D is correct!")
else:
    print("einsum in 2D is incorrect")

# with np.tensordot:
test_tensordot = np.tensordot(X, Y, (0, 0))
print(test_tensordot.shape)
if np.alltrue(test_tensordot == test_forloop):
    print("tensordot is correct!")
else:
    print("tensordot is incorrect")

# with np.einsum in 2D:
test_tensordot_2D = np.tensordot(X, Y_2D, (0, 0))
print(test_tensordot_2D.shape)
if np.alltrue(test_tensordot_2D == test_forloop_2D):
    print("tensordot in 2D is correct!")
else:
    print("tensordot in 2D is incorrect")

# with cov
test_cov = cov(X, Y)
print(test_cov.shape)
if np.alltrue(test_cov == test_forloop):
    print("cov is correct!")
else:
    print("cov is incorrect")

# with cov in 2D
test_cov_2D = cov(X, Y_2D)
print(test_cov_2D.shape)
if np.alltrue(test_cov_2D == test_forloop_2D):
    print("cov_2D is correct!")
else:
    print("cov_2D is incorrect")
