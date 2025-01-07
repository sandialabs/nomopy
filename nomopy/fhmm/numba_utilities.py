import os
import sys
import numpy as np
from numba import njit
from numba import config

if os.getenv("PYTEST_DISABLE_JIT") == "1":
       config.DISABLE_JIT = True


@njit(fastmath=True)
def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1/(np.power(2*np.pi, size/2) * np.power(det, 1/2))
        x_mu = x - mu
        inv = np.linalg.inv(sigma)
        result = np.power(np.e, -0.5 * (np.dot(np.dot(x_mu, inv), np.transpose(x_mu))))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

@njit(fastmath=True)
def jit_py(T, d, k, o, x, realizations, W, C):
    # eps = 1E-15  #np.finfo(np.float32).eps
    eps = 1E-32

    py = np.zeros(shape=(T, k**d))
    s_t = np.zeros(shape=(d, k))
    for t in range(T):
        for i in range(realizations.shape[1]):
            # Setup the hidden state realization.
            s_t *= 0
            for idx_d in range(d):
                s_t[idx_d, realizations[idx_d, i]] = 1

            # Get the Y_t probability given this realization slice
            y_mu = np.zeros(shape=o)
            for idx_d in range(d):
                w = W[idx_d, :, :].copy()
                s = s_t[idx_d, :].copy()
                y_mu += np.dot(w, s)
                # y_mu += np.dot(W[idx_d, :, :], s_t[idx_d, :])
            py[t, i] = norm_pdf_multivariate(x[t, :], y_mu, C)

    py += eps

    return py

@njit(fastmath=True)
def jit_forward(T, d, k, realizations, py, A, pi):
    # eps = 1E-15 #np.finfo(np.float32).eps
    eps = 0

    # Initialize alpha
    alpha = np.ones(shape=(T, k**d))
    for i in range(realizations.shape[1]):
        joint_pi = 1
        for idx_d in range(d):
            joint_pi *= pi[idx_d, realizations[idx_d, i]]

        alpha[0, i] = joint_pi * py[0, i] + eps

    # Normalizing constant
    c = np.zeros(shape=(T,))

    c[0] = alpha[0, :].sum()
    alpha[0, :] /= c[0]

    for t in range(1, T):
        for j in range(realizations.shape[1]):
            prob_j = np.ones(shape=realizations.shape[1])
            for idx_d in range(d):
                temp = np.exp(A)[idx_d, realizations[idx_d, j], :]
                prob_j *= temp[realizations[idx_d, :]]
            alpha[t, j] = np.sum(alpha[t-1] * prob_j * py[t, j]) + eps
        c[t] = alpha[t, :].sum()
        alpha[t, :] /= c[t]

    return c, alpha

@njit(fastmath=True)
def jit_backward(T, d, k, realizations, py, A):
    # eps = 1E-15 #np.finfo(np.float32).eps
    eps = 0

    # Normalizing constant
    c = np.zeros(shape=(T,))

    beta = np.ones(shape=(T, k**d))
    c[T-1] = beta[T-1, :].sum()
    beta[T-1, :] /= c[T-1]

    t_range = list(range(1, T))[::-1]

    # Recursion
    for t in t_range:
        for j in range(0, realizations.shape[1]):
            prob_j = np.ones(shape=realizations.shape[1])
            for idx_d in range(d):
                temp = np.exp(A)[idx_d, :, realizations[idx_d, j]]
                prob_j *= temp[realizations[idx_d, :]]
            beta[t-1, j] = np.sum(beta[t] * prob_j * py[t, :]) + eps
        c[t-1] = beta[t-1, :].sum()
        beta[t-1, :] /= c[t-1]

    return c, beta

@njit(fastmath=True)
def jit_exact(T, d, k, realizations, k_contrib, py, A, pi):
    eps = 0 #np.finfo(np.float32).eps

    _, alpha = jit_forward(T, d, k, realizations, py, A, pi)  # Don't need the normalization
    _, beta = jit_backward(T, d, k, realizations, py, A)  # Don't need the normalization

    gamma = alpha * beta
    norm = gamma.sum(axis=1).reshape(T, -1)
    gamma /= norm

    ########
    # s_exp

    # Assign elements via for loop...
    s_exp = eps * np.ones(shape=(T, d, k))
    for t in range(T):
        for idx_d in range(d):
            for idx_k in range(k):
                indices = k_contrib[idx_d, idx_k, :]
                temp = gamma[t, :]
                s_exp[t, idx_d, idx_k] = np.sum(temp[indices])

    #########
    # ss_exp

    # Assign elements via for loop...
    ss_exp = eps * np.ones(shape=(T, d, d, k, k))
    for t in range(T):
        for d1 in range(d):
            for d2 in range(d):
                for k1 in range(k):
                    for k2 in range(k):
                        indices = np.array(list(set(k_contrib[d1, k1, :]) & set(k_contrib[d2, k2, :]))).astype(np.int64)
                        temp = gamma[t, :]
                        ss_exp[t, d1, d2, k1, k2] = np.sum(temp[indices])

    ############
    # sstm1_exp

    # Construct the product of transition probabilities
    psstm1 = np.ones(shape=(realizations.shape[1],
                                realizations.shape[1]))
    for t_i in range(realizations.shape[1]):
        for idx_d in range(d):
            temp = np.exp(A)[idx_d, realizations[idx_d, t_i], :]
            psstm1[t_i, :] *= temp[realizations[idx_d, :]]

    # Assign elements via for loop...
    sstm1_exp = eps * np.ones(shape=(T, d, k, k))
    for t in range(1, T):
        norm_t = eps
        for idx_d in range(d):
            for k1 in range(k):
                for k2 in range(k):
                    t_indices = k_contrib[idx_d, k1, :]
                    tm1_indices = k_contrib[idx_d, k2, :]

                    for t_idx in t_indices:
                        nmba_alpha = alpha[t-1, :]
                        nmba_psstm1 = psstm1[t_idx, :]
                        sstm1_exp[t, idx_d, k1, k2] += np.sum(nmba_alpha[tm1_indices]
                                                                * nmba_psstm1[tm1_indices]
                                                                * py[t, t_idx]
                                                                * beta[t, t_idx])

                    # Running sum for normalization
                    norm_t += sstm1_exp[t, idx_d, k1, k2]

        sstm1_exp[t, :, :, :] /= (norm_t/d)

    return s_exp, ss_exp, sstm1_exp
