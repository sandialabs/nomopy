""" Machinery for computing the Hessian of the loglikelihood with
respect to model parameters.
"""
from itertools import product

import numpy as np
from joblib import Parallel, delayed


class Hessian():
    """Machinery for computing the Hessian of the loglikelihood

    Parameters
    ----------
    T : int, (Z+)
        The length of each sequence.
    D : int, (Z+)
        The number of hidden vectors, at each time step.
    K : int, (Z+)
        The length of the hidden vectors -- number of states.
    O : int, (Z+)
        The length of the output vector.
    x: np.array of shape (T, o)
        Time series example for which to calculate hessian of log likelihood.
    W : numpy.array
    A : numpy.array
    C : numpy.array
    pi : numpy.array
    C_inv : numpy.array
    realizations: numpy.array
    k_contrib: numpy.array
        Array version of the k_contrib mapping.
    py : numpy.array
    """

    def __init__(self, T, D, O, K, x, W, A, C, pi, C_inv, realizations,
                 k_contrib, py):
        self.T = T
        self.D = D
        self.O = O
        self.K = K
        self.x = x

        self.W = W
        self.A = A
        self.C = C
        self.pi = pi
        self.C_inv = C_inv

        self.realizations = realizations
        self.k_contrib = k_contrib
        self.py = py


    def dpy_dw(self, d, o, k):
        dpydw = np.zeros(shape=(self.T, self.K**self.D))
        s_t = np.zeros(shape=(self.D, self.K))
        for t in range(self.T):
            for r in range(self.realizations.shape[1]):
                # Setup the hidden state realization.
                s_t *= 0
                for idx_d in range(self.D):
                    s_t[idx_d, self.realizations[idx_d, r]] = 1

                d_constraint = s_t[d, -1] if d != 0 else 0

                y_err = self.x[t, :] - np.einsum('dok,dk', self.W, s_t)
                sCyWs = (s_t[d, k] - d_constraint) * self.C_inv[o, :].dot(y_err)

                dpydw[t, r] = self.py[t, r] * sCyWs

        return dpydw


    def d2py_dwdw(self, e, p, l, d, o, k):
        d2pydwdw = np.zeros(shape=(self.T, self.K**self.D))
        s_t = np.zeros(shape=(self.D, self.K))
        for t in range(self.T):
            for r in range(self.realizations.shape[1]):
                # Setup the hidden state realization.
                s_t *= 0
                for idx_d in range(self.D):
                    s_t[idx_d, self.realizations[idx_d, r]] = 1

                d_constraint = s_t[d, -1] if d != 0 else 0
                e_constraint = s_t[e, -1] if e != 0 else 0

                y_err = self.x[t, :] - np.einsum('dok,dk', self.W, s_t)
                sCyWs1 = (s_t[e, l] - e_constraint) * self.C_inv[p, :].dot(y_err)
                sCyWs2 = (s_t[d, k] - d_constraint) * self.C_inv[o, :].dot(y_err)
                sCs = (s_t[e, l] - e_constraint) * self.C_inv[p, o] * (s_t[d, k] - d_constraint)

                d2pydwdw[t, r] = self.py[t, r] * (sCyWs1 * sCyWs2 - sCs)

        return d2pydwdw

    def dpy_dc(self, i, j):
        dpydc = np.zeros(shape=(self.T, self.K**self.D))
        s_t = np.zeros(shape=(self.D, self.K))
        for t in range(self.T):
            for r in range(self.realizations.shape[1]):
                # Setup the hidden state realization.
                s_t *= 0
                for idx_d in range(self.D):
                    s_t[idx_d, self.realizations[idx_d, r]] = 1

                y_err = self.x[t, :] - np.einsum('dok,dk', self.W, s_t)
                yCCy = y_err.dot(self.C_inv[:, i]) * self.C_inv[:, j].dot(y_err) - self.C_inv[i, j]

                dpydc[t, r] = 1/2 * self.py[t, r] * yCCy

        return dpydc

    def d2py_dcdc(self, l, m, i, j):
        d2pydcdc = np.zeros(shape=(self.T, self.K**self.D))
        s_t = np.zeros(shape=(self.D, self.K))
        for t in range(self.T):
            for r in range(self.realizations.shape[1]):
                # Setup the hidden state realization.
                s_t *= 0
                for idx_d in range(self.D):
                    s_t[idx_d, self.realizations[idx_d, r]] = 1

                y_err = self.x[t, :] - np.einsum('dok,dk', self.W, s_t)
                yCCy1 = y_err.dot(self.C_inv[:, l]) * self.C_inv[:, m].dot(y_err) - self.C_inv[l, m]
                yCCy2 = y_err.dot(self.C_inv[:, i]) * self.C_inv[:, j].dot(y_err) - self.C_inv[i, j]
                yCCCy1 = y_err.dot(self.C_inv[:, l]) * self.C_inv[i, m] * self.C_inv[:, j].dot(y_err)
                yCCCy2 = y_err.dot(self.C_inv[:, i]) * self.C_inv[:, l].dot(y_err) * self.C_inv[j, m]

                d2pydcdc[t, r] = 1/4 * self.py[t, r] * yCCy1 * yCCy2 \
                              + 1/2 * self.py[t, r] * (self.C_inv[i, l] * self.C_inv[j, m] - yCCCy1 - yCCCy2)

        return d2pydcdc


    def d2py_dwdc(self, d, o, k, i, j):
        d2pydwdc = np.zeros(shape=(self.T, self.K**self.D))
        s_t = np.zeros(shape=(self.D, self.K))
        for t in range(self.T):
            for r in range(self.realizations.shape[1]):
                # Setup the hidden state realization.
                s_t *= 0
                for idx_d in range(self.D):
                    s_t[idx_d, self.realizations[idx_d, r]] = 1

                d_constraint = s_t[d, -1] if d != 0 else 0

                y_err = self.x[t, :] - np.einsum('dok,dk', self.W, s_t)
                sCyWs = (s_t[d, k] - d_constraint) * self.C_inv[o, :].dot(y_err)
                dpydw = self.py[t, r] * sCyWs

                yCCy = y_err.dot(self.C_inv[:, i]) * self.C_inv[:, j].dot(y_err)
                sCCy = (s_t[d, k] - d_constraint) * self.C_inv[o, i] * self.C_inv[:, j].dot(y_err)
                yCCs = y_err.dot(self.C_inv[:, i]) * self.C_inv[o, j] * (s_t[d, k] - d_constraint)

                d2pydwdc[t, r] = 1/2 * dpydw * (yCCy - self.C_inv[i, j]) \
                                 - 1/2 * self.py[t, r] * (sCCy + yCCs)

        return d2pydwdc


    def hessian_WW(self, e, p, l, d, o, k):
        dpydw1 = self.dpy_dw(d, o, k)
        dpydw2 = self.dpy_dw(e, p, l)
        d2pydwdw = self.d2py_dwdw(e, p, l, d, o, k)

        # Initialize alpha
        alpha = np.ones(shape=(self.T, self.K**self.D))
        dalphadw1 = np.ones(shape=(self.T, self.K**self.D))
        dalphadw2 = np.ones(shape=(self.T, self.K**self.D))
        d2alphadwdw = np.ones(shape=(self.T, self.K**self.D))
        for i in range(self.realizations.shape[1]):
            joint_pi = 1
            for idx_d in range(self.D):
                joint_pi *= self.pi[idx_d, self.realizations[idx_d, i]]

            alpha[0, i] = joint_pi * self.py[0, i]
            dalphadw1[0, i] = joint_pi * dpydw1[0, i]
            dalphadw2[0, i] = joint_pi * dpydw2[0, i]
            d2alphadwdw[0, i] = joint_pi * d2pydwdw[0, i]

        # Normalizing constant
        c = np.zeros(shape=(self.T,))
        dcdw1 = np.zeros(shape=(self.T,))
        dcdw2 = np.zeros(shape=(self.T,))
        d2cdwdw = np.zeros(shape=(self.T,))

        c[0] = alpha[0, :].sum()
        alpha[0, :] /= c[0]
        dcdw1[0] = dalphadw1[0, :].sum()
        dalphadw1[0, :] /= c[0]
        dcdw2[0] = dalphadw2[0, :].sum()
        dalphadw2[0, :] /= c[0]
        d2cdwdw[0] = d2alphadwdw[0, :].sum()
        d2alphadwdw[0, :] /= c[0]

        for t in range(1, self.T):
            for r in range(self.realizations.shape[1]):
                prob_r = np.ones(shape=self.realizations.shape[1])
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    prob_r *= temp[self.realizations[idx_d, :]]
                alpha[t, r] = np.sum(alpha[t-1] * prob_r * self.py[t, r])
                dalphadw1[t, r] = np.sum(alpha[t-1] * prob_r * dpydw1[t, r]) \
                                  + np.sum((dalphadw1[t-1] - dcdw1[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                dalphadw2[t, r] = np.sum(alpha[t-1] * prob_r * dpydw2[t, r]) \
                                  + np.sum((dalphadw2[t-1] - dcdw2[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                d2alphadwdw[t, r] = np.sum((dalphadw2[t-1] - alpha[t-1] * dcdw2[t-1]/c[t-1]) * prob_r * dpydw1[t, r]) \
                                    + np.sum(alpha[t-1] * prob_r * d2pydwdw[t, r]) \
                                    + np.sum((d2alphadwdw[t-1] + 2 * dcdw2[t-1]/c[t-1] * dcdw1[t-1]/c[t-1] * alpha[t-1] - d2cdwdw[t-1]/c[t-1] * alpha[t-1] - dcdw1[t-1]/c[t-1] * dalphadw2[t-1] - dcdw2[t-1]/c[t-1] * dalphadw1[t-1]) * prob_r * self.py[t, r]) \
                                    + np.sum((dalphadw1[t-1] - alpha[t-1] * dcdw1[t-1]/c[t-1]) * prob_r * dpydw2[t, r])

            c[t] = alpha[t, :].sum()
            alpha[t, :] /= c[t]
            dcdw1[t] = dalphadw1[t, :].sum()
            dalphadw1[t, :] /= c[t]
            dcdw2[t] = dalphadw2[t, :].sum()
            dalphadw2[t, :] /= c[t]
            d2cdwdw[t] = d2alphadwdw[t, :].sum()
            d2alphadwdw[t, :] /= c[t]

        hessian_ww = 0
        for t in range(self.T):
            hessian_ww += -1/c[t]**2 * dcdw2[t] * dcdw1[t] + 1/c[t] * d2cdwdw[t]

        return hessian_ww


    def hessian_AA(self, e, m, n, d, k, l):
        # Initialize alpha
        alpha = np.ones(shape=(self.T, self.K**self.D))
        dalphadA1 = np.ones(shape=(self.T, self.K**self.D))
        dalphadA2 = np.ones(shape=(self.T, self.K**self.D))
        d2alphadAdA = np.ones(shape=(self.T, self.K**self.D))
        for i in range(self.realizations.shape[1]):
            joint_pi = 1
            for idx_d in range(self.D):
                joint_pi *= self.pi[idx_d, self.realizations[idx_d, i]]

            alpha[0, i] = joint_pi * self.py[0, i]
            dalphadA1[0, i] = 0
            dalphadA2[0, i] = 0
            d2alphadAdA[0, i] = 0

        # Normalizing constant
        c = np.zeros(shape=(self.T,))
        dcdA1 = np.zeros(shape=(self.T,))
        dcdA2 = np.zeros(shape=(self.T,))
        d2cdAdA = np.zeros(shape=(self.T,))

        c[0] = alpha[0, :].sum()
        alpha[0, :] /= c[0]
        dcdA1[0] = dalphadA1[0, :].sum()
        dalphadA1[0, :] /= c[0]
        dcdA2[0] = dalphadA2[0, :].sum()
        dalphadA2[0, :] /= c[0]
        d2cdAdA[0] = d2alphadAdA[0, :].sum()
        d2alphadAdA[0, :] /= c[0]

        for t in range(1, self.T):
            for r in range(self.realizations.shape[1]):
                prob_r = np.ones(shape=self.realizations.shape[1])
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    prob_r *= temp[self.realizations[idx_d, :]]

                # First derivative of prob_r
                dprob_rdA1 = np.ones(shape=self.realizations.shape[1])
                k_indices = self.k_contrib[d, k]
                d_Km1_indices = self.k_contrib[d, self.K-1]
                l_indices_contrib = np.zeros(shape=self.realizations.shape[1])
                l_indices_contrib[self.k_contrib[d, l]] = 1
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    if idx_d == d:
                        if r in k_indices:
                            dprob_rdA1 *= temp[self.realizations[idx_d, :]] * l_indices_contrib  # Only keep terms that have d, k, l
                        elif r in d_Km1_indices:
                            temp2 = np.exp(self.A)[idx_d, k, :]
                            dprob_rdA1 *= -temp2[self.realizations[idx_d, :]] * l_indices_contrib  # Only keep terms that have d, k, l
                        else:
                            dprob_rdA1 *= 0
                            break
                    else:
                        dprob_rdA1 *= temp[self.realizations[idx_d, :]]

                # First derivative of prob_r
                dprob_rdA2 = np.ones(shape=self.realizations.shape[1])
                m_indices = self.k_contrib[e, m]
                e_Km1_indices = self.k_contrib[e, self.K-1]
                n_indices_contrib = np.zeros(shape=self.realizations.shape[1])
                n_indices_contrib[self.k_contrib[e, n]] = 1
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    if idx_d == e:
                        if r in m_indices:
                            dprob_rdA2 *= temp[self.realizations[idx_d, :]] * n_indices_contrib  # Only keep terms that have e, m, n
                        elif r in e_Km1_indices:
                            temp2 = np.exp(self.A)[idx_d, m, :]
                            dprob_rdA2 *= -temp2[self.realizations[idx_d, :]] * n_indices_contrib  # Only keep terms that have e, m, n
                        else:
                            dprob_rdA2 *= 0  # The derivative is zero
                            break
                    else:
                        dprob_rdA2 *= temp[self.realizations[idx_d, :]]

                # Second derivative of prob_r
                d2prob_rdAdA = np.ones(shape=self.realizations.shape[1])
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    if idx_d == d and d != e:
                        if r in k_indices:
                            d2prob_rdAdA *= temp[self.realizations[idx_d, :]] * l_indices_contrib  # Only keep terms that have d, k, l
                        elif r in d_Km1_indices:
                            temp2 = np.exp(self.A)[idx_d, k, :]
                            d2prob_rdAdA *= -temp2[self.realizations[idx_d, :]] * l_indices_contrib  # Only keep terms that have e, m, n
                        else:
                            d2prob_rdAdA *= 0
                            break

                    elif idx_d == e and d != e:
                        if r in m_indices:
                            d2prob_rdAdA *= temp[self.realizations[idx_d, :]] * n_indices_contrib  # Only keep terms that have e, m, n
                        elif r in e_Km1_indices:
                            temp2 = np.exp(self.A)[idx_d, m, :]
                            d2prob_rdAdA *= -temp2[self.realizations[idx_d, :]] * n_indices_contrib  # Only keep terms that have e, m, n
                        else:
                            d2prob_rdAdA *= 0  # The derivative is zero
                            break

                    elif idx_d == e and d == e:
                        if r in k_indices and r in m_indices:
                            d2prob_rdAdA *= temp[self.realizations[idx_d, :]] * l_indices_contrib * n_indices_contrib  # Only keep terms that have d, k, l
                        elif r in d_Km1_indices and r in e_Km1_indices:
                            temp2 = np.exp(self.A)[idx_d, m, :]
                            d2prob_rdAdA *= -temp2[self.realizations[idx_d, :]] * l_indices_contrib * n_indices_contrib
                        else:
                            d2prob_rdAdA *= 0
                            break
                    else:
                        assert idx_d != d and idx_d != e
                        d2prob_rdAdA *= temp[self.realizations[idx_d, :]]

                # Now finally the alpha recursion contributions
                alpha[t, r] = np.sum(alpha[t-1] * prob_r * self.py[t, r])
                dalphadA1[t, r] = np.sum(alpha[t-1] * dprob_rdA1 * self.py[t, r]) \
                                  + np.sum((dalphadA1[t-1] - dcdA1[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                dalphadA2[t, r] = np.sum(alpha[t-1] * dprob_rdA2 * self.py[t, r]) \
                                  + np.sum((dalphadA2[t-1] - dcdA2[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                d2alphadAdA[t, r] = np.sum((dalphadA2[t-1] - alpha[t-1] * dcdA2[t-1]/c[t-1]) * dprob_rdA1 * self.py[t, r]) \
                                    + np.sum(alpha[t-1] * d2prob_rdAdA * self.py[t, r]) \
                                    + np.sum((d2alphadAdA[t-1] + 2 * dcdA2[t-1]/c[t-1] * dcdA1[t-1]/c[t-1] * alpha[t-1] - d2cdAdA[t-1]/c[t-1] * alpha[t-1] - dcdA1[t-1]/c[t-1] * dalphadA2[t-1] - dcdA2[t-1]/c[t-1] * dalphadA1[t-1]) * prob_r * self.py[t, r]) \
                                    + np.sum((dalphadA1[t-1] - alpha[t-1] * dcdA1[t-1]/c[t-1]) * dprob_rdA2 * self.py[t, r])

            c[t] = alpha[t, :].sum()
            alpha[t, :] /= c[t]
            dcdA1[t] = dalphadA1[t, :].sum()
            dalphadA1[t, :] /= c[t]
            dcdA2[t] = dalphadA2[t, :].sum()
            dalphadA2[t, :] /= c[t]
            d2cdAdA[t] = d2alphadAdA[t, :].sum()
            d2alphadAdA[t, :] /= c[t]

        hessian_aa = 0
        for t in range(self.T):
            hessian_aa += -1/c[t]**2 * dcdA2[t] * dcdA1[t] + 1/c[t] * d2cdAdA[t]

        return hessian_aa


    def hessian_WA(self, d, o, k, e, m, n):
        dpydw = self.dpy_dw(d, o, k)

        # Initialize alpha
        alpha = np.ones(shape=(self.T, self.K**self.D))
        dalphadw = np.ones(shape=(self.T, self.K**self.D))
        dalphadA = np.ones(shape=(self.T, self.K**self.D))
        d2alphadwdA = np.ones(shape=(self.T, self.K**self.D))
        for i in range(self.realizations.shape[1]):
            joint_pi = 1
            for idx_d in range(self.D):
                joint_pi *= self.pi[idx_d, self.realizations[idx_d, i]]

            alpha[0, i] = joint_pi * self.py[0, i]
            dalphadw[0, i] = joint_pi * dpydw[0, i]
            dalphadA[0, i] = 0
            d2alphadwdA[0, i] = 0

        # Normalizing constant
        c = np.zeros(shape=(self.T,))
        dcdw = np.zeros(shape=(self.T,))
        dcdA = np.zeros(shape=(self.T,))
        d2cdwdA = np.zeros(shape=(self.T,))

        c[0] = alpha[0, :].sum()
        alpha[0, :] /= c[0]
        dcdw[0] = dalphadw[0, :].sum()
        dalphadw[0, :] /= c[0]
        dcdA[0] = dalphadA[0, :].sum()
        dalphadA[0, :] /= c[0]
        d2cdwdA[0] = d2alphadwdA[0, :].sum()
        d2alphadwdA[0, :] /= c[0]

        for t in range(1, self.T):
            for r in range(self.realizations.shape[1]):
                prob_r = np.ones(shape=self.realizations.shape[1])
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    prob_r *= temp[self.realizations[idx_d, :]]

                # First derivative of prob_r
                dprob_rdA = np.ones(shape=self.realizations.shape[1])
                m_indices = self.k_contrib[e, m]
                e_Km1_indices = self.k_contrib[e, self.K-1]
                n_indices_contrib = np.zeros(shape=self.realizations.shape[1])
                n_indices_contrib[self.k_contrib[e, n]] = 1
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    if idx_d == e:
                        if r in m_indices:
                            dprob_rdA *= temp[self.realizations[idx_d, :]] * n_indices_contrib  # Only keep terms that have e, m, n
                        elif r in e_Km1_indices:
                            temp2 = np.exp(self.A)[idx_d, m, :]
                            dprob_rdA *= -temp2[self.realizations[idx_d, :]] * n_indices_contrib  # Only keep terms that have e, m, n
                        else:
                            dprob_rdA *= 0  # The derivative is zero
                            break
                    else:
                        dprob_rdA *= temp[self.realizations[idx_d, :]]

                # Now finally the alpha recursion contributions
                alpha[t, r] = np.sum(alpha[t-1] * prob_r * self.py[t, r])
                dalphadw[t, r] = np.sum(alpha[t-1] * prob_r * dpydw[t, r]) \
                                 + np.sum((dalphadw[t-1] - dcdw[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                dalphadA[t, r] = np.sum(alpha[t-1] * dprob_rdA * self.py[t, r]) \
                                 + np.sum((dalphadA[t-1] - dcdA[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                d2alphadwdA[t, r] = np.sum((dalphadA[t-1] - alpha[t-1] * dcdA[t-1]/c[t-1]) * prob_r * dpydw[t, r]) \
                                    + np.sum(alpha[t-1] * dprob_rdA * dpydw[t, r]) \
                                    + np.sum((dalphadw[t-1] - dcdw[t-1]/c[t-1] * alpha[t-1]) * dprob_rdA * self.py[t, r]) \
                                    + np.sum((d2alphadwdA[t-1] + 2 * dcdA[t-1]/c[t-1] * dcdw[t-1]/c[t-1] * alpha[t-1] - d2cdwdA[t-1]/c[t-1] * alpha[t-1] - dcdw[t-1]/c[t-1] * dalphadA[t-1] - dcdA[t-1]/c[t-1] * dalphadw[t-1]) * prob_r * self.py[t, r])

            c[t] = alpha[t, :].sum()
            alpha[t, :] /= c[t]
            dcdw[t] = dalphadw[t, :].sum()
            dalphadw[t, :] /= c[t]
            dcdA[t] = dalphadA[t, :].sum()
            dalphadA[t, :] /= c[t]
            d2cdwdA[t] = d2alphadwdA[t, :].sum()
            d2alphadwdA[t, :] /= c[t]

        hessian_wa = 0
        for t in range(self.T):
            hessian_wa += -1/c[t]**2 * dcdA[t] * dcdw[t] + 1/c[t] * d2cdwdA[t]

        return hessian_wa


    def hessian_WC(self, d, o, k, p, q):
        dpydw = self.dpy_dw(d, o, k)
        dpydC = self.dpy_dc(p, q)
        d2pydwdC = self.d2py_dwdc(d, o, k, p, q)

        # Initialize alpha
        alpha = np.ones(shape=(self.T, self.K**self.D))
        dalphadw = np.ones(shape=(self.T, self.K**self.D))
        dalphadC = np.ones(shape=(self.T, self.K**self.D))
        d2alphadwdC = np.ones(shape=(self.T, self.K**self.D))
        for i in range(self.realizations.shape[1]):
            joint_pi = 1
            for idx_d in range(self.D):
                joint_pi *= self.pi[idx_d, self.realizations[idx_d, i]]

            alpha[0, i] = joint_pi * self.py[0, i]
            dalphadw[0, i] = joint_pi * dpydw[0, i]
            dalphadC[0, i] = joint_pi * dpydC[0, i]
            d2alphadwdC[0, i] = joint_pi * d2pydwdC[0, i]

        # Normalizing constant
        c = np.zeros(shape=(self.T,))
        dcdw = np.zeros(shape=(self.T,))
        dcdC = np.zeros(shape=(self.T,))
        d2cdwdC = np.zeros(shape=(self.T,))

        c[0] = alpha[0, :].sum()
        alpha[0, :] /= c[0]
        dcdw[0] = dalphadw[0, :].sum()
        dalphadw[0, :] /= c[0]
        dcdC[0] = dalphadC[0, :].sum()
        dalphadC[0, :] /= c[0]
        d2cdwdC[0] = d2alphadwdC[0, :].sum()
        d2alphadwdC[0, :] /= c[0]

        for t in range(1, self.T):
            for r in range(self.realizations.shape[1]):
                prob_r = np.ones(shape=self.realizations.shape[1])
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    prob_r *= temp[self.realizations[idx_d, :]]

                # Now finally the alpha recursion contributions
                alpha[t, r] = np.sum(alpha[t-1] * prob_r * self.py[t, r])
                dalphadw[t, r] = np.sum(alpha[t-1] * prob_r * dpydw[t, r]) \
                                 + np.sum((dalphadw[t-1] - dcdw[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                dalphadC[t, r] = np.sum(alpha[t-1] * prob_r * dpydC[t, r]) \
                                 + np.sum((dalphadC[t-1] - dcdC[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                d2alphadwdC[t, r] = np.sum((dalphadC[t-1] - alpha[t-1] * dcdC[t-1]/c[t-1]) * prob_r * dpydw[t, r]) \
                                    + np.sum(alpha[t-1] * prob_r * d2pydwdC[t, r]) \
                                    + np.sum((dalphadw[t-1] - dcdw[t-1]/c[t-1] * alpha[t-1]) * prob_r * dpydC[t, r]) \
                                    + np.sum((d2alphadwdC[t-1] + 2 * dcdC[t-1]/c[t-1] * dcdw[t-1]/c[t-1] * alpha[t-1] - d2cdwdC[t-1]/c[t-1] * alpha[t-1] - dcdw[t-1]/c[t-1] * dalphadC[t-1] - dcdC[t-1]/c[t-1] * dalphadw[t-1]) * prob_r * self.py[t, r])

            c[t] = alpha[t, :].sum()
            alpha[t, :] /= c[t]
            dcdw[t] = dalphadw[t, :].sum()
            dalphadw[t, :] /= c[t]
            dcdC[t] = dalphadC[t, :].sum()
            dalphadC[t, :] /= c[t]
            d2cdwdC[t] = d2alphadwdC[t, :].sum()
            d2alphadwdC[t, :] /= c[t]

        hessian_wc = 0
        for t in range(self.T):
            hessian_wc += -1/c[t]**2 * dcdC[t] * dcdw[t] + 1/c[t] * d2cdwdC[t]

        return hessian_wc


    def hessian_WPi(self, d, o, k, e, l):
        dpydw = self.dpy_dw(d, o, k)

        # Initialize alpha
        alpha = np.ones(shape=(self.T, self.K**self.D))
        dalphadw = np.ones(shape=(self.T, self.K**self.D))
        dalphadpi = np.ones(shape=(self.T, self.K**self.D))
        d2alphadwdpi = np.ones(shape=(self.T, self.K**self.D))
        for i in range(self.realizations.shape[1]):
            joint_pi = 1
            for idx_d in range(self.D):
                joint_pi *= self.pi[idx_d, self.realizations[idx_d, i]]

            # First derivative
            djoint_dpi = 1
            for idx_d in range(self.D):
                if idx_d == e:
                    if self.realizations[idx_d, i] == l:
                        continue  # take derivative by continuing, excluding from product
                    elif self.realizations[idx_d, i] == self.K-1:  # Last state by convention
                        djoint_dpi *= -1
                        continue
                    else:
                        djoint_dpi *= 0
                djoint_dpi *= self.pi[idx_d, self.realizations[idx_d, i]]

            alpha[0, i] = joint_pi * self.py[0, i]
            dalphadw[0, i] = joint_pi * dpydw[0, i]
            dalphadpi[0, i] = djoint_dpi * self.py[0, i]
            d2alphadwdpi[0, i] = djoint_dpi * dpydw[0, i]

        # Normalizing constant
        c = np.zeros(shape=(self.T,))
        dcdw = np.zeros(shape=(self.T,))
        dcdpi = np.zeros(shape=(self.T,))
        d2cdwdpi = np.zeros(shape=(self.T,))

        c[0] = alpha[0, :].sum()
        alpha[0, :] /= c[0]
        dcdw[0] = dalphadw[0, :].sum()
        dalphadw[0, :] /= c[0]
        dcdpi[0] = dalphadpi[0, :].sum()
        dalphadpi[0, :] /= c[0]
        d2cdwdpi[0] = d2alphadwdpi[0, :].sum()
        d2alphadwdpi[0, :] /= c[0]

        for t in range(1, self.T):
            for r in range(self.realizations.shape[1]):
                prob_r = np.ones(shape=self.realizations.shape[1])
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    prob_r *= temp[self.realizations[idx_d, :]]

                # Now finally the alpha recursion contributions
                alpha[t, r] = np.sum(alpha[t-1] * prob_r * self.py[t, r])
                dalphadw[t, r] = np.sum(alpha[t-1] * prob_r * dpydw[t, r]) \
                                 + np.sum((dalphadw[t-1] - dcdw[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                dalphadpi[t, r] = np.sum((dalphadpi[t-1] - dcdpi[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                d2alphadwdpi[t, r] = np.sum((dalphadpi[t-1] - alpha[t-1] * dcdpi[t-1]/c[t-1]) * prob_r * dpydw[t, r]) \
                                    + np.sum((d2alphadwdpi[t-1] + 2 * dcdpi[t-1]/c[t-1] * dcdw[t-1]/c[t-1] * alpha[t-1] - d2cdwdpi[t-1]/c[t-1] * alpha[t-1] - dcdw[t-1]/c[t-1] * dalphadpi[t-1] - dcdpi[t-1]/c[t-1] * dalphadw[t-1]) * prob_r * self.py[t, r])

            c[t] = alpha[t, :].sum()
            alpha[t, :] /= c[t]
            dcdw[t] = dalphadw[t, :].sum()
            dalphadw[t, :] /= c[t]
            dcdpi[t] = dalphadpi[t, :].sum()
            dalphadpi[t, :] /= c[t]
            d2cdwdpi[t] = d2alphadwdpi[t, :].sum()
            d2alphadwdpi[t, :] /= c[t]

        hessian_wpi = 0
        for t in range(self.T):
            hessian_wpi += -1/c[t]**2 * dcdpi[t] * dcdw[t] + 1/c[t] * d2cdwdpi[t]

        return hessian_wpi


    def hessian_AC(self, e, m, n, p, q):
        dpydC = self.dpy_dc(p, q)

        # Initialize alpha
        alpha = np.ones(shape=(self.T, self.K**self.D))
        dalphadA = np.ones(shape=(self.T, self.K**self.D))
        dalphadC = np.ones(shape=(self.T, self.K**self.D))
        d2alphadAdC = np.ones(shape=(self.T, self.K**self.D))
        for i in range(self.realizations.shape[1]):
            joint_pi = 1
            for idx_d in range(self.D):
                joint_pi *= self.pi[idx_d, self.realizations[idx_d, i]]

            alpha[0, i] = joint_pi * self.py[0, i]
            dalphadA[0, i] = 0
            dalphadC[0, i] = joint_pi * dpydC[0, i]
            d2alphadAdC[0, i] = 0

        # Normalizing constant
        c = np.zeros(shape=(self.T,))
        dcdA = np.zeros(shape=(self.T,))
        dcdC = np.zeros(shape=(self.T,))
        d2cdAdC = np.zeros(shape=(self.T,))

        c[0] = alpha[0, :].sum()
        alpha[0, :] /= c[0]
        dcdA[0] = dalphadA[0, :].sum()
        dalphadA[0, :] /= c[0]
        dcdC[0] = dalphadC[0, :].sum()
        dalphadC[0, :] /= c[0]
        d2cdAdC[0] = d2alphadAdC[0, :].sum()
        d2alphadAdC[0, :] /= c[0]

        for t in range(1, self.T):
            for r in range(self.realizations.shape[1]):
                prob_r = np.ones(shape=self.realizations.shape[1])
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    prob_r *= temp[self.realizations[idx_d, :]]

                # First derivative of prob_r
                dprob_rdA = np.ones(shape=self.realizations.shape[1])
                m_indices = self.k_contrib[e, m]
                e_Km1_indices = self.k_contrib[e, self.K-1]
                n_indices_contrib = np.zeros(shape=self.realizations.shape[1])
                n_indices_contrib[self.k_contrib[e, n]] = 1
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    if idx_d == e:
                        if r in m_indices:
                            dprob_rdA *= temp[self.realizations[idx_d, :]] * n_indices_contrib  # Only keep terms that have e, m, n
                        elif r in e_Km1_indices:
                            temp2 = np.exp(self.A)[idx_d, m, :]
                            dprob_rdA *= -temp2[self.realizations[idx_d, :]] * n_indices_contrib  # Only keep terms that have e, m, n
                        else:
                            dprob_rdA *= 0  # The derivative is zero
                            break
                    else:
                        dprob_rdA *= temp[self.realizations[idx_d, :]]

                # Now finally the alpha recursion contributions
                alpha[t, r] = np.sum(alpha[t-1] * prob_r * self.py[t, r])
                dalphadC[t, r] = np.sum(alpha[t-1] * prob_r * dpydC[t, r]) \
                                 + np.sum((dalphadC[t-1] - dcdC[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                dalphadA[t, r] = np.sum(alpha[t-1] * dprob_rdA * self.py[t, r]) \
                                 + np.sum((dalphadA[t-1] - dcdA[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                d2alphadAdC[t, r] = np.sum((dalphadA[t-1] - alpha[t-1] * dcdA[t-1]/c[t-1]) * prob_r * dpydC[t, r]) \
                                    + np.sum(alpha[t-1] * dprob_rdA * dpydC[t, r]) \
                                    + np.sum((dalphadC[t-1] - dcdC[t-1]/c[t-1] * alpha[t-1]) * dprob_rdA * self.py[t, r]) \
                                    + np.sum((d2alphadAdC[t-1] + 2 * dcdA[t-1]/c[t-1] * dcdC[t-1]/c[t-1] * alpha[t-1] - d2cdAdC[t-1]/c[t-1] * alpha[t-1] - dcdC[t-1]/c[t-1] * dalphadA[t-1] - dcdA[t-1]/c[t-1] * dalphadC[t-1]) * prob_r * self.py[t, r])

            c[t] = alpha[t, :].sum()
            alpha[t, :] /= c[t]
            dcdC[t] = dalphadC[t, :].sum()
            dalphadC[t, :] /= c[t]
            dcdA[t] = dalphadA[t, :].sum()
            dalphadA[t, :] /= c[t]
            d2cdAdC[t] = d2alphadAdC[t, :].sum()
            d2alphadAdC[t, :] /= c[t]

        hessian_ac = 0
        for t in range(self.T):
            hessian_ac += -1/c[t]**2 * dcdA[t] * dcdC[t] + 1/c[t] * d2cdAdC[t]

        return hessian_ac


    def hessian_APi(self, e, m, n, d, k):
        # Initialize alpha
        alpha = np.ones(shape=(self.T, self.K**self.D))
        dalphadA = np.ones(shape=(self.T, self.K**self.D))
        dalphadpi = np.ones(shape=(self.T, self.K**self.D))
        d2alphadAdpi = np.ones(shape=(self.T, self.K**self.D))
        for i in range(self.realizations.shape[1]):
            joint_pi = 1
            for idx_d in range(self.D):
                joint_pi *= self.pi[idx_d, self.realizations[idx_d, i]]

            # First derivative
            djoint_dpi = 1
            for idx_d in range(self.D):
                if idx_d == d:
                    if self.realizations[idx_d, i] == k:
                        continue  # take derivative by continuing, excluding from product
                    elif self.realizations[idx_d, i] == self.K-1:  # Last state by convention
                        djoint_dpi *= -1
                        continue
                    else:
                        djoint_dpi *= 0
                djoint_dpi *= self.pi[idx_d, self.realizations[idx_d, i]]

            alpha[0, i] = joint_pi * self.py[0, i]
            dalphadA[0, i] = 0
            dalphadpi[0, i] = djoint_dpi * self.py[0, i]
            d2alphadAdpi[0, i] = 0

        # Normalizing constant
        c = np.zeros(shape=(self.T,))
        dcdA = np.zeros(shape=(self.T,))
        dcdpi = np.zeros(shape=(self.T,))
        d2cdAdpi = np.zeros(shape=(self.T,))

        c[0] = alpha[0, :].sum()
        alpha[0, :] /= c[0]
        dcdA[0] = dalphadA[0, :].sum()
        dalphadA[0, :] /= c[0]
        dcdpi[0] = dalphadpi[0, :].sum()
        dalphadpi[0, :] /= c[0]
        d2cdAdpi[0] = d2alphadAdpi[0, :].sum()
        d2alphadAdpi[0, :] /= c[0]

        for t in range(1, self.T):
            for r in range(self.realizations.shape[1]):
                prob_r = np.ones(shape=self.realizations.shape[1])
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    prob_r *= temp[self.realizations[idx_d, :]]

                # First derivative of prob_r
                dprob_rdA = np.ones(shape=self.realizations.shape[1])
                m_indices = self.k_contrib[e, m]
                e_Km1_indices = self.k_contrib[e, self.K-1]
                n_indices_contrib = np.zeros(shape=self.realizations.shape[1])
                n_indices_contrib[self.k_contrib[e, n]] = 1
                assert len(set(m_indices) & set(e_Km1_indices)) == 0
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    if idx_d == e:
                        if r in m_indices:
                            dprob_rdA *= temp[self.realizations[idx_d, :]] * n_indices_contrib  # Only keep terms that have e, m, n
                        elif r in e_Km1_indices:
                            temp2 = np.exp(self.A)[idx_d, m, :]
                            dprob_rdA *= -temp2[self.realizations[idx_d, :]] * n_indices_contrib  # Only keep terms that have e, m, n
                        else:
                            dprob_rdA *= 0  # The derivative is zero
                            break
                    else:
                        dprob_rdA *= temp[self.realizations[idx_d, :]]

                # Now finally the alpha recursion contributions
                alpha[t, r] = np.sum(alpha[t-1] * prob_r * self.py[t, r])
                dalphadA[t, r] = np.sum(alpha[t-1] * dprob_rdA * self.py[t, r]) \
                                 + np.sum((dalphadA[t-1] - dcdA[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                dalphadpi[t, r] = np.sum((dalphadpi[t-1] - dcdpi[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                d2alphadAdpi[t, r] = np.sum((dalphadpi[t-1] - alpha[t-1] * dcdpi[t-1]/c[t-1]) * dprob_rdA * self.py[t, r]) \
                                     + np.sum((d2alphadAdpi[t-1] + 2 * dcdpi[t-1]/c[t-1] * dcdA[t-1]/c[t-1] * alpha[t-1] - d2cdAdpi[t-1]/c[t-1] * alpha[t-1] - dcdA[t-1]/c[t-1] * dalphadpi[t-1] - dcdpi[t-1]/c[t-1] * dalphadA[t-1]) * prob_r * self.py[t, r])

            c[t] = alpha[t, :].sum()
            alpha[t, :] /= c[t]
            dcdA[t] = dalphadA[t, :].sum()
            dalphadA[t, :] /= c[t]
            dcdpi[t] = dalphadpi[t, :].sum()
            dalphadpi[t, :] /= c[t]
            d2cdAdpi[t] = d2alphadAdpi[t, :].sum()
            d2alphadAdpi[t, :] /= c[t]

        hessian_api = 0
        for t in range(self.T):
            hessian_api += -1/c[t]**2 * dcdpi[t] * dcdA[t] + 1/c[t] * d2cdAdpi[t]

        return hessian_api


    def hessian_CC(self, n, o, p, q):
        dpydC1 = self.dpy_dc(n, o)
        dpydC2 = self.dpy_dc(p, q)
        d2pydCdC = self.d2py_dcdc(n, o, p, q)

        # Initialize alpha
        alpha = np.ones(shape=(self.T, self.K**self.D))
        dalphadC1 = np.ones(shape=(self.T, self.K**self.D))
        dalphadC2 = np.ones(shape=(self.T, self.K**self.D))
        d2alphadCdC = np.ones(shape=(self.T, self.K**self.D))
        for i in range(self.realizations.shape[1]):
            joint_pi = 1
            for idx_d in range(self.D):
                joint_pi *= self.pi[idx_d, self.realizations[idx_d, i]]

            alpha[0, i] = joint_pi * self.py[0, i]
            dalphadC1[0, i] = joint_pi * dpydC1[0, i]
            dalphadC2[0, i] = joint_pi * dpydC2[0, i]
            d2alphadCdC[0, i] = joint_pi * d2pydCdC[0, i]

        # Normalizing constant
        c = np.zeros(shape=(self.T,))
        dcdC1 = np.zeros(shape=(self.T,))
        dcdC2 = np.zeros(shape=(self.T,))
        d2cdCdC = np.zeros(shape=(self.T,))

        c[0] = alpha[0, :].sum()
        alpha[0, :] /= c[0]
        dcdC1[0] = dalphadC1[0, :].sum()
        dalphadC1[0, :] /= c[0]
        dcdC2[0] = dalphadC2[0, :].sum()
        dalphadC2[0, :] /= c[0]
        d2cdCdC[0] = d2alphadCdC[0, :].sum()
        d2alphadCdC[0, :] /= c[0]

        for t in range(1, self.T):
            for r in range(self.realizations.shape[1]):
                prob_r = np.ones(shape=self.realizations.shape[1])
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    prob_r *= temp[self.realizations[idx_d, :]]

                # Now finally the alpha recursion contributions
                alpha[t, r] = np.sum(alpha[t-1] * prob_r * self.py[t, r])
                dalphadC1[t, r] = np.sum(alpha[t-1] * prob_r * dpydC1[t, r]) \
                                 + np.sum((dalphadC1[t-1] - dcdC1[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                dalphadC2[t, r] = np.sum(alpha[t-1] * prob_r * dpydC2[t, r]) \
                                 + np.sum((dalphadC2[t-1] - dcdC2[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                d2alphadCdC[t, r] = np.sum((dalphadC2[t-1] - alpha[t-1] * dcdC2[t-1]/c[t-1]) * prob_r * dpydC1[t, r]) \
                                    + np.sum(alpha[t-1] * prob_r * d2pydCdC[t, r]) \
                                    + np.sum((dalphadC1[t-1] - dcdC1[t-1]/c[t-1] * alpha[t-1]) * prob_r * dpydC2[t, r]) \
                                    + np.sum((d2alphadCdC[t-1] + 2 * dcdC2[t-1]/c[t-1] * dcdC1[t-1]/c[t-1] * alpha[t-1] - d2cdCdC[t-1]/c[t-1] * alpha[t-1] - dcdC1[t-1]/c[t-1] * dalphadC2[t-1] - dcdC2[t-1]/c[t-1] * dalphadC1[t-1]) * prob_r * self.py[t, r])

            c[t] = alpha[t, :].sum()
            alpha[t, :] /= c[t]
            dcdC1[t] = dalphadC1[t, :].sum()
            dalphadC1[t, :] /= c[t]
            dcdC2[t] = dalphadC2[t, :].sum()
            dalphadC2[t, :] /= c[t]
            d2cdCdC[t] = d2alphadCdC[t, :].sum()
            d2alphadCdC[t, :] /= c[t]

        hessian_cc = 0
        for t in range(self.T):
            hessian_cc += -1/c[t]**2 * dcdC2[t] * dcdC1[t] + 1/c[t] * d2cdCdC[t]

        return hessian_cc


    def hessian_CPi(self, n, o, e, l):
        dpydC = self.dpy_dc(n, o)

        # Initialize alpha
        alpha = np.ones(shape=(self.T, self.K**self.D))
        dalphadC = np.ones(shape=(self.T, self.K**self.D))
        dalphadpi = np.ones(shape=(self.T, self.K**self.D))
        d2alphadCdpi = np.ones(shape=(self.T, self.K**self.D))
        for i in range(self.realizations.shape[1]):
            joint_pi = 1
            for idx_d in range(self.D):
                joint_pi *= self.pi[idx_d, self.realizations[idx_d, i]]

            # First derivative
            djoint_dpi = 1
            for idx_d in range(self.D):
                if idx_d == e:
                    if self.realizations[idx_d, i] == l:
                        continue  # take derivative by continuing, excluding from product
                    elif self.realizations[idx_d, i] == self.K-1:  # Last state by convention
                        djoint_dpi *= -1
                        continue
                    else:
                        djoint_dpi *= 0
                djoint_dpi *= self.pi[idx_d, self.realizations[idx_d, i]]

            alpha[0, i] = joint_pi * self.py[0, i]
            dalphadC[0, i] = joint_pi * dpydC[0, i]
            dalphadpi[0, i] = djoint_dpi * self.py[0, i]
            d2alphadCdpi[0, i] = djoint_dpi * dpydC[0, i]

        # Normalizing constant
        c = np.zeros(shape=(self.T,))
        dcdC = np.zeros(shape=(self.T,))
        dcdpi = np.zeros(shape=(self.T,))
        d2cdCdpi = np.zeros(shape=(self.T,))

        c[0] = alpha[0, :].sum()
        alpha[0, :] /= c[0]
        dcdC[0] = dalphadC[0, :].sum()
        dalphadC[0, :] /= c[0]
        dcdpi[0] = dalphadpi[0, :].sum()
        dalphadpi[0, :] /= c[0]
        d2cdCdpi[0] = d2alphadCdpi[0, :].sum()
        d2alphadCdpi[0, :] /= c[0]

        for t in range(1, self.T):
            for r in range(self.realizations.shape[1]):
                prob_r = np.ones(shape=self.realizations.shape[1])
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    prob_r *= temp[self.realizations[idx_d, :]]

                # Now finally the alpha recursion contributions
                alpha[t, r] = np.sum(alpha[t-1] * prob_r * self.py[t, r])
                dalphadC[t, r] = np.sum(alpha[t-1] * prob_r * dpydC[t, r]) \
                                 + np.sum((dalphadC[t-1] - dcdC[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                dalphadpi[t, r] = np.sum((dalphadpi[t-1] - dcdpi[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                d2alphadCdpi[t, r] = np.sum((dalphadpi[t-1] - alpha[t-1] * dcdpi[t-1]/c[t-1]) * prob_r * dpydC[t, r]) \
                                    + np.sum((d2alphadCdpi[t-1] + 2 * dcdpi[t-1]/c[t-1] * dcdC[t-1]/c[t-1] * alpha[t-1] - d2cdCdpi[t-1]/c[t-1] * alpha[t-1] - dcdC[t-1]/c[t-1] * dalphadpi[t-1] - dcdpi[t-1]/c[t-1] * dalphadC[t-1]) * prob_r * self.py[t, r])

            c[t] = alpha[t, :].sum()
            alpha[t, :] /= c[t]
            dcdC[t] = dalphadC[t, :].sum()
            dalphadC[t, :] /= c[t]
            dcdpi[t] = dalphadpi[t, :].sum()
            dalphadpi[t, :] /= c[t]
            d2cdCdpi[t] = d2alphadCdpi[t, :].sum()
            d2alphadCdpi[t, :] /= c[t]

        hessian_cpi = 0
        for t in range(self.T):
            hessian_cpi += -1/c[t]**2 * dcdpi[t] * dcdC[t] + 1/c[t] * d2cdCdpi[t]

        return hessian_cpi


    def hessian_PiPi(self, e, l, d, k):
        # Initialize alpha
        alpha = np.ones(shape=(self.T, self.K**self.D))
        dalphadpi1 = np.ones(shape=(self.T, self.K**self.D))
        dalphadpi2 = np.ones(shape=(self.T, self.K**self.D))
        d2alphadpidpi = np.ones(shape=(self.T, self.K**self.D))
        for i in range(self.realizations.shape[1]):
            joint_pi = 1
            for idx_d in range(self.D):
                joint_pi *= self.pi[idx_d, self.realizations[idx_d, i]]

            # First derivative
            djoint_dpi1 = 1
            for idx_d in range(self.D):
                if idx_d == e:
                    if self.realizations[idx_d, i] == l:
                        continue  # take derivative by continuing, excluding from product
                    elif self.realizations[idx_d, i] == self.K-1:  # Last state by convention
                        djoint_dpi1 *= -1
                        continue
                    else:
                        djoint_dpi1 *= 0
                djoint_dpi1 *= self.pi[idx_d, self.realizations[idx_d, i]]

            # First derivative
            djoint_dpi2 = 1
            for idx_d in range(self.D):
                if idx_d == d:
                    if self.realizations[idx_d, i] == k:
                        continue  # take derivative by continuing, excluding from product
                    elif self.realizations[idx_d, i] == self.K-1:  # Last state by convention
                        djoint_dpi2 *= -1
                        continue
                    else:
                        djoint_dpi2 *= 0
                djoint_dpi2 *= self.pi[idx_d, self.realizations[idx_d, i]]

            # Second derivative
            d2joint_dpidpi = 1
            for idx_d in range(self.D):
                if idx_d == d and d != e:
                    if self.realizations[idx_d, i] == k:
                        mult_factor = 1
                    elif self.realizations[idx_d, i] == self.K-1:  # Last state by convention
                        mult_factor = -1
                    else:
                        d2joint_dpidpi *= 0
                        break
                elif idx_d == e and d != e:
                    if self.realizations[idx_d, i] == l:
                        mult_factor = 1
                    elif self.realizations[idx_d, i] == self.K-1:  # Last state by convention
                        mult_factor = -1
                    else:
                        d2joint_dpidpi *= 0
                        break
                elif idx_d == d and d == e:
                    d2joint_dpidpi = 0  # These second chain derivatives will be zero
                    break
                else:
                    assert idx_d != d and idx_d != e
                    mult_factor = self.pi[idx_d, self.realizations[idx_d, i]]

                d2joint_dpidpi *= mult_factor

            alpha[0, i] = joint_pi * self.py[0, i]
            dalphadpi1[0, i] = djoint_dpi1 * self.py[0, i]
            dalphadpi2[0, i] = djoint_dpi2 * self.py[0, i]
            d2alphadpidpi[0, i] = d2joint_dpidpi * self.py[0, i]

        # Normalizing constant
        c = np.zeros(shape=(self.T,))
        dcdpi1 = np.zeros(shape=(self.T,))
        dcdpi2 = np.zeros(shape=(self.T,))
        d2cdpidpi = np.zeros(shape=(self.T,))

        c[0] = alpha[0, :].sum()
        alpha[0, :] /= c[0]
        dcdpi1[0] = dalphadpi1[0, :].sum()
        dalphadpi1[0, :] /= c[0]
        dcdpi2[0] = dalphadpi2[0, :].sum()
        dalphadpi2[0, :] /= c[0]
        d2cdpidpi[0] = d2alphadpidpi[0, :].sum()
        d2alphadpidpi[0, :] /= c[0]

        for t in range(1, self.T):
            for r in range(self.realizations.shape[1]):
                prob_r = np.ones(shape=self.realizations.shape[1])
                for idx_d in range(self.D):
                    temp = np.exp(self.A)[idx_d, self.realizations[idx_d, r], :]
                    prob_r *= temp[self.realizations[idx_d, :]]

                # Now finally the alpha recursion contributions
                alpha[t, r] = np.sum(alpha[t-1] * prob_r * self.py[t, r])
                dalphadpi1[t, r] = np.sum((dalphadpi1[t-1] - dcdpi1[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                dalphadpi2[t, r] = np.sum((dalphadpi2[t-1] - dcdpi2[t-1]/c[t-1] * alpha[t-1]) * prob_r * self.py[t, r])
                d2alphadpidpi[t, r] = np.sum((d2alphadpidpi[t-1] + 2 * dcdpi2[t-1]/c[t-1] * dcdpi1[t-1]/c[t-1] * alpha[t-1] - d2cdpidpi[t-1]/c[t-1] * alpha[t-1] - dcdpi1[t-1]/c[t-1] * dalphadpi2[t-1] - dcdpi2[t-1]/c[t-1] * dalphadpi1[t-1]) * prob_r * self.py[t, r])

            c[t] = alpha[t, :].sum()
            alpha[t, :] /= c[t]
            dcdpi1[t] = dalphadpi1[t, :].sum()
            dalphadpi1[t, :] /= c[t]
            dcdpi2[t] = dalphadpi2[t, :].sum()
            dalphadpi2[t, :] /= c[t]
            d2cdpidpi[t] = d2alphadpidpi[t, :].sum()
            d2alphadpidpi[t, :] /= c[t]

        hessian_pipi = 0
        for t in range(self.T):
            hessian_pipi += -1/c[t]**2 * dcdpi2[t] * dcdpi1[t] + 1/c[t] * d2cdpidpi[t]

        return hessian_pipi


    def mat2block_with_offset(self, i, j):
        """ Takes the flattened indices i, j and calculates the 'block'
        number which is 0 for W, 1 for A, 2 for C, and 3 for pi.  Then
        also calculates the within block offset -- if pi has 3 independent
        degrees of freedom and the i,j element corresponds to the 2nd, then
        we have block 3, offset 1 (zero indexed).

        Parameters
        ----------
        i: int
            Row of flattened index.
        j: int
            Column of flattened index.

        Returns
        -------
        rblock, cblock, roffset, coffset: tuple
            rblock is 0 for W, 1 for A, 2 for C, and 3 for pi block of the
            concatenated independent parameters. roffset calculates the
            within block parameter offset.
        """
        startW = 0
        endW = startW + ( self.D*self.O*self.K - (self.D-1)*self.O )

        startA = endW
        endA = startA + self.D*(self.K-1)*self.K

        startC = endA
        endC = startC + self.O*self.O

        startpi = endC
        endpi = startpi + self.D*(self.K-1)

        if i < endW:
            rblock = 0
        elif i < endA:
            rblock = 1
        elif i < endC:
            rblock = 2
        elif i < endpi:
            rblock = 3

        if j < endW:
            cblock = 0
        elif j < endA:
            cblock = 1
        elif j < endC:
            cblock = 2
        elif j < endpi:
            cblock = 3

        if rblock == 0:
            roffset = i
        elif rblock == 1:
            roffset = i - endW
        elif rblock == 2:
            roffset = i - endA
        elif rblock == 3:
            roffset = i - endC

        if cblock == 0:
            coffset = j
        elif cblock == 1:
            coffset = j - endW
        elif cblock == 2:
            coffset = j - endA
        elif cblock == 3:
            coffset = j - endC

        return rblock, cblock, roffset, coffset


    def hessian_element(self, i, j):
        r""" Calculate particular (i, j) hessian element, where the index
        corresponds to the unravelled independent elements of W, A, C,
        and pi, in that particular order. For example (0, 0) will be
        the first independent element of canonically transformed W upon
        unravelling. This function lends to easy parallelization.

        Parameters
        ----------
        i: int, (Z+)
            Row selection of square hessian matrix.
        j: int, (Z+)
            Column selection of square hessian matrix.

        Returns
        -------
        val: float
            The value of the Hessian element (i, j).

        Notes
        -----
        Complexity :math:`\mathcal{O}(Tdk^{d+1})`.
        """
        rblock, cblock, roffset, coffset = self.mat2block_with_offset(i, j)

        if rblock == 0 and cblock == 0:
            W_indices = list(product(*[list(range(self.W.shape[i])) for i in range(len(self.W.shape))]))
            W_indices_constrained = [i for i in W_indices if not (i[2] == self.K-1 and i[0] > 0)]

            matr_idx = W_indices_constrained[roffset]
            matc_idx = W_indices_constrained[coffset]

            val = self.hessian_WW(*matr_idx, *matc_idx)
        elif rblock == 0 and cblock == 1:
            W_indices = list(product(*[list(range(self.W.shape[i])) for i in range(len(self.W.shape))]))
            W_indices_constrained = [i for i in W_indices if not (i[2] == self.K-1 and i[0] > 0)]

            A_indices = list(product(*[list(range(self.A.shape[i])) for i in range(len(self.A.shape))]))
            A_indices_constrained = [i for i in A_indices if not i[1] == self.K-1]

            matr_idx = W_indices_constrained[roffset]
            matc_idx = A_indices_constrained[coffset]

            val = self.hessian_WA(*matr_idx, *matc_idx)
        elif rblock == 0 and cblock == 2:
            W_indices = list(product(*[list(range(self.W.shape[i])) for i in range(len(self.W.shape))]))
            W_indices_constrained = [i for i in W_indices if not (i[2] == self.K-1 and i[0] > 0)]
            C_indices = list(product(*[list(range(self.C.shape[i])) for i in range(len(self.C.shape))]))

            matr_idx = W_indices_constrained[roffset]
            matc_idx = C_indices[coffset]

            val = self.hessian_WC(*matr_idx, *matc_idx)
        elif rblock == 0 and cblock == 3:
            W_indices = list(product(*[list(range(self.W.shape[i])) for i in range(len(self.W.shape))]))
            W_indices_constrained = [i for i in W_indices if not (i[2] == self.K-1 and i[0] > 0)]
            pi_indices = list(product(*[list(range(self.pi.shape[i])) for i in range(len(self.pi.shape))]))
            pi_indices_constrained = [i for i in pi_indices if not i[1] == self.K-1]

            matr_idx = W_indices_constrained[roffset]
            matc_idx = pi_indices_constrained[coffset]

            val = self.hessian_WPi(*matr_idx, *matc_idx)
        elif rblock == 1 and cblock == 1:
            A_indices = list(product(*[list(range(self.A.shape[i])) for i in range(len(self.A.shape))]))
            A_indices_constrained = [i for i in A_indices if not i[1] == self.K-1]

            matr_idx = A_indices_constrained[roffset]
            matc_idx = A_indices_constrained[coffset]

            val = self.hessian_AA(*matr_idx, *matc_idx)
        elif rblock == 1 and cblock == 2:
            A_indices = list(product(*[list(range(self.A.shape[i])) for i in range(len(self.A.shape))]))
            A_indices_constrained = [i for i in A_indices if not i[1] == self.K-1]
            C_indices = list(product(*[list(range(self.C.shape[i])) for i in range(len(self.C.shape))]))

            matr_idx = A_indices_constrained[roffset]
            matc_idx = C_indices[coffset]

            val = self.hessian_AC(*matr_idx, *matc_idx)
        elif rblock == 1 and cblock == 3:
            A_indices = list(product(*[list(range(self.A.shape[i])) for i in range(len(self.A.shape))]))
            A_indices_constrained = [i for i in A_indices if not i[1] == self.K-1]
            pi_indices = list(product(*[list(range(self.pi.shape[i])) for i in range(len(self.pi.shape))]))
            pi_indices_constrained = [i for i in pi_indices if not i[1] ==
                                      self.K-1]

            matr_idx = A_indices_constrained[roffset]
            matc_idx = pi_indices_constrained[coffset]

            val = self.hessian_APi(*matr_idx, *matc_idx)
        elif rblock == 2 and cblock == 2:
            C_indices = list(product(*[list(range(self.C.shape[i])) for i in range(len(self.C.shape))]))

            matr_idx = C_indices[roffset]
            matc_idx = C_indices[coffset]

            val = self.hessian_CC(*matr_idx, *matc_idx)
        elif rblock == 2 and cblock == 3:
            C_indices = list(product(*[list(range(self.C.shape[i])) for i in range(len(self.C.shape))]))
            pi_indices = list(product(*[list(range(self.pi.shape[i])) for i in range(len(self.pi.shape))]))
            pi_indices_constrained = [i for i in pi_indices if not i[1] ==
                                      self.K-1]

            matr_idx = C_indices[roffset]
            matc_idx = pi_indices_constrained[coffset]

            val = self.hessian_CPi(*matr_idx, *matc_idx)
        elif rblock == 3 and cblock == 3:
            pi_indices = list(product(*[list(range(self.pi.shape[i])) for i in range(len(self.pi.shape))]))
            pi_indices_constrained = [i for i in pi_indices if not i[1] ==
                                      self.K-1]

            matr_idx = pi_indices_constrained[roffset]
            matc_idx = pi_indices_constrained[coffset]

            val = self.hessian_PiPi(*matr_idx, *matc_idx)

        return val


    def hessian(self):
        r""" Evaluate the Hessian of the log likelihood. Upper triangle
        is evaluated and then the rest are filled in using the transpose.

        Returns
        -------
        hess: np.array of shape (Np, Np) [see Notes below]

        Notes
        -----
        The number of independent W elements is :math:`dok-(d-1)o`; for A
        this is :math:`d(k-1)k`; for C this is :math:`o^{2}`; and for pi
        this is :math:`d(k-1)`. So,

        .. math:: N_p = dok - (d-1)o + d(k-1)k + o^{2} + d(k-1)

        Complexity :math:`\mathcal{O}(Tdk^{d+1})` if computing all
        :math:`N_p(N_p +1)/2` elements in parallel.
        """
        dim = self.D*self.O*self.K - (self.D-1)*self.O + self.D*(self.K-1)*self.K + self.O*self.O + self.D*(self.K-1)

        hess = np.zeros(shape=(dim, dim))
        futures = []
        for idx_i in range(dim):
            for idx_j in range(dim):
                if idx_j < idx_i: continue
                hessian_val = delayed(self.hessian_element)(idx_i, idx_j)
                futures.append(hessian_val)

        hessian_flat = Parallel(n_jobs=-1)(futures)

        hessian_gen = (h for h in hessian_flat)
        for idx_i in range(dim):
            for idx_j in range(dim):
                if idx_j < idx_i: continue
                hess[idx_i, idx_j] = next(hessian_gen)

        hess = hess + hess.T - np.diag(hess.diagonal())

        return hess
