import datetime
import os
import pickle
import warnings
from copy import deepcopy
from itertools import product
from collections import defaultdict
from functools import lru_cache

import numpy as np
from numpy.linalg import LinAlgError
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed

from .convergence_monitor import ConvergenceMonitor
from .exact_hessian import Hessian
from .numba_utilities import norm_pdf_multivariate
from .numba_utilities import jit_py
from .numba_utilities import jit_exact
from .numba_utilities import jit_forward


class FHMM(BaseEstimator):
    """Factorial Hidden Markov Model (FHMM) implementation.

    Parameters
    ----------
    T : int, (Z+)
        The length of each sequence.
    d : int, (Z+)
        The number of hidden vectors, at each time step.
    k : int, (Z+)
        The length of the hidden vectors -- number of states.
    o : int, (Z+)
        The length of the output vector.
    n_restarts : int, (Z+)
        Number of full model restarts, in search of the global optimum.
    em_max_iter : int, (Z+)
        Maximum number of cycles through E-M steps.
    em_log_likelihood_tol : float, default=1E-8
        The tolerance level to discern one likelihood value to the next.
    em_log_likelihood_count : int, (Z+)
        Number of log likelihood values without change (according to
        `em_log_likelihood_tol`) indicating convergence.
    e_step_retries : int, (Z+)
        Number of random restarts of (applicable) E method.
    method : str, ('gibbs', 'mean_field', 'sva', 'exact')
        Selecting the method for expectation maximization. Options
        are 'gibbs' for Gibbs sampling, 'mean_field' for
        using mean field estimation (or completely factorized approximation), and
        'sva' to use the Structured Variational Approximation (SVA), and
        'exact' for the exact solve (very slow -- do not use for large parameter values!).
    gibbs_max_iter : int, (Z+)
        Number of states sampled within Gibbs E-step.
    mean_field_max_iter : int, (Z+)
        Maximum number of mean field updates.  Once reached, will
        exit without necessarily meeting KLD tolerance.
    mean_field_kld_tol : float, (R+)
        Tolerance for change in KLD between mean field iterations.
    sva_max_iter : int, (Z+)
        Maximum number of Structured Variational Approximation (SVA) updates.  Once reached, will
        exit without necessarily meeting KLD tolerance.
    sva_kld_tol : float, (R+)
        Tolerance for change in KLD between SVA iterations.
    stochastic_training : bool
        Whether or not to use stochastic training -- random and decaying jostling of fit parameters while learning.
    stochastic_lr : float
        Roughly the size of the random excursions in fit parameters.
    zero_probability : float, (R+)
        Numerical cutoff indicating zero probability (not strictly zero).
    W_init : numpy.array, None
        Initialize the starting W weight matrix (shape (T, d, k)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial W.
    A_init : numpy.array, None
        Initialize the starting A transition matrix (shape=(d, k, k)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial A.
    C_init : numpy.array, None
        Initialize the starting C covariance matrix (shape=(o, o)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial C.
    pi_init : numpy.array, None
        Initialize the starting pi initial state distribution matrix (shape=(d, k)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial pi.
    W_fixed : numpy.array, None
        Set equal to the true W weight matrix (shape (T, d, k)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update W.
    A_fixed : numpy.array, None
        Set equal to the true A transition matrix (shape=(d, k, k)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update A.
    C_fixed : numpy.array, None
        Set equal to the true C covariance matrix (shape=(o, o)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update C.
    pi_fixed : numpy.array, None
        Set equal to the true pi initial state distribution matrix (shape=(d, k)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update pi.
    save_directory : str, None
        Location to save data for each iteration (if defined)
    save_x : bool, False
        Save X if save_directory is defined
    verbose : bool, True
        Print progress and possibly other indicators of algorithm state.

    Attributes
    ----------
    hessian_: np.array, None
        The Hessian for the sample calculated.
    dW: np.array, None
        Standard error for W.
    dA: np.array, None
        Standard error for A.
    dC: np.array, None
        Standard error for C.
    dpi: np.array, None
        Standard error for pi.
    s_exps: list, None
        List of s expectation values for each time series. Length is number
        of examples.
    ss_exps: list, None
        List of ss expectation values for each time series. Length is number
        of examples.
    sstm1_exps:
        List of sstm1 expectation values for each time series. Length is number
        of examples.
    n_samples: int, None
        Number of time series examples.
    X: np.array, None
        Time series data for fitting.  Shape = (N, T, o).
    h: np.array, None
        SVA method attribute. Store approximate probability for
        observable given state, for each sample at each time.
        shape=(num_samples, T, d, k)
    states: np.array(gibbs_max_iter, T, d, k), None
        Gibbs method attribute. Stores the trace of hidden states.
    ps: np.array(gibbs_max_iter, T, d, k), None
        Gibbs method attribute. Stores the trace of conditional probabilities.
    s: np.array(T, d, k), None
        Gibbs method attribute.  Stores the current iteration's hidden states.
    it: int, None
        EM loop iteration tracking for stochastic training decay.

    Examples
    --------
    >>> # Fitting 20 sequences of length 10, output vector of len 4.
    >>> parameters = {'T': 10,  # Total number of time steps
    ...               'd': 4,   # Number of hidden vectors at each time 't'
    ...               'k': 8,   # Number of possible states for each hidden vector 's'
    ...               'o': 4,   # Output vector 'y' dimension
    ...              }
    >>> W, A, C, pi = FHMM.generate_random_model_params(*parameters)
    >>> X = FHMM(*parameters, W_fixed=W, A_fixed=A, C_fixed=C, pi_fixed=pi).generate(20, parameters['T'])
    >>> X.shape  # (number of samples, length of sequence, output vector length)
    (20, 10, 4)
    >>> fhmm = FHMM()
    >>> fhmm.fit(X)

    >>> # Testing out W estimation
    >>> fhmm = FHMM(W_fixed=W)
    >>> fhmm.estimate_W()
    """

    def __init__(self,
                 T=10,
                 d=4,
                 k=8,
                 o=4,
                 n_restarts=5,
                 em_max_iter=10,
                 em_log_likelihood_tol=1E-8,
                 em_log_likelihood_count=4,
                 e_step_retries=0,
                 method='sva',
                 gibbs_max_iter=200,
                 mean_field_max_iter=10,
                 mean_field_kld_tol=1E-8,
                 sva_max_iter=10,
                 sva_kld_tol=1E-8,
                 stochastic_training=False,
                 stochastic_lr=0.1,
                 zero_probability=1E-8,
                 W_init=None,
                 A_init=None,
                 C_init=None,
                 pi_init=None,
                 W_fixed=None,
                 A_fixed=None,
                 C_fixed=None,
                 pi_fixed=None,
                 save_directory=None,
                 save_x=False,
                 verbose=True):

        self.T = T
        self.d = d
        self.k = k
        self.o = o

        self.n_restarts = n_restarts

        self.em_max_iter = em_max_iter
        self.em_log_likelihood_tol = em_log_likelihood_tol
        self.em_log_likelihood_count = em_log_likelihood_count
        self.e_step_retries = e_step_retries
        self.method = method

        self.gibbs_max_iter = gibbs_max_iter
        self.mean_field_max_iter = mean_field_max_iter
        self.mean_field_kld_tol = mean_field_kld_tol
        self.sva_max_iter = sva_max_iter
        self.sva_kld_tol = sva_kld_tol

        self.zero_probability = zero_probability

        self.W_init = W_init
        self.A_init = A_init
        self.C_init = C_init
        self.pi_init = pi_init

        # Fixing the true values of these parameters, or None
        self.W_fixed = W_fixed
        self.A_fixed = A_fixed
        self.C_fixed = C_fixed
        self.pi_fixed = pi_fixed

        self.save_directory = save_directory
        self.save_x = save_x

        self.stochastic_training = stochastic_training
        self.stochastic_lr = stochastic_lr

        self.verbose = verbose

        self.hessian_samples = []
        self.hessian_ = None
        self.dW = None
        self.dA = None
        self.dC = None
        self.dpi = None

        self.s_exps = []
        self.ss_exps = []
        self.sstm1_exps = []

        self.n_samples = None
        self.X = None

        # SVA observable likelihood
        self.h = None

        # Iteration tracking for stochastic annealing
        self.it = 0

        self.initialize_parameters()

        self.convergence_monitor = ConvergenceMonitor(
                                          0,
                                          self.expected_complete_log_likelihood,
                                          self.em_log_likelihood_count,
                                          self.em_log_likelihood_tol,
                                          verbose=self.verbose)

    def initialize_parameters(self):
        self._initialize_W()
        self._initialize_A()
        self._initialize_C()
        self._initialize_pi()

        if self.method == 'gibbs':
            self._initialize_hidden_states()
            self._initialize_trace_variables()

    def _initialize_W(self):
        if self.W_init is not None:
            self.W = self.W_init.copy()
        else:
            self.W = np.random.uniform(0, 1, size=self.k*self.o*self.d)\
                              .reshape(self.d, self.o, self.k)

    def _initialize_A(self):
        if self.A_init is not None:
            self.A = self.A_init.copy()
        else:
            self.A = self.generate_random_A(self.d, self.k, self.k)

    def _initialize_C(self):
        if self.C_init is not None:
            self.C = self.C_init.copy()
        else:
            self.C = self.generate_random_C(self.o)

    def _initialize_pi(self):
        if self.pi_init is not None:
            self.pi = self.pi_init.copy()
        else:
            self.pi = self.generate_random_pi(self.d, self.k)

    def _initialize_hidden_states(self):
        self.s = np.ndarray(shape=(self.T, self.d, self.k))
        for t in range(self.T):
            self._set_hidden_states_at_t(t)

    def _initialize_trace_variables(self):
        self.states = np.ndarray(shape=(self.gibbs_max_iter,
                                        self.T,
                                        self.d,
                                        self.k))
        self.ps = np.ndarray(shape=(self.gibbs_max_iter,
                                    self.T,
                                    self.d,
                                    self.k))


    def _set_hidden_states_at_t(self, t):
        for d in range(self.d):
            idx = self._draw_hidden_state_index_from_probability(d, t)
            self.s[t, d, :] = 0
            self.s[t, d, idx] = 1

    def _draw_hidden_state_index_from_probability(self, d, t):
        p = self.pi[d] if t == 0 else np.exp(self.A[d]).dot(self.s[t-1, d, :])
        idx = np.random.choice(range(self.k), p=p)
        return idx

    @staticmethod
    def generate_random_A(d, k1, k2):
        A = np.random.uniform(1E-9, 1, size=k1*k2*d).reshape(d, k1, k2)
        A /= A.sum(axis=1, keepdims=True)
        A = np.log(A)
        return A

    @staticmethod
    def generate_random_C(o):
        mat = np.random.uniform(0, 1, size=o**2).reshape(o, o)
        cov = (mat.T).dot(mat)
        return cov

    @staticmethod
    def generate_random_pi(d, k):
        pi = np.random.uniform(1E-9, 1, size=d*k).reshape(d, k)
        pi /= pi.sum(axis=1, keepdims=True)
        return pi

    @property
    def W(self):
        return self.W_

    @W.setter
    def W(self, W):
        if self.W_fixed is not None:
            self.W_ = self.W_fixed.copy()
        else:
            self.W_ = W

    @property
    def A(self):
        return self.A_

    @A.setter
    def A(self, A):
        if self.A_fixed is not None:
            self.A_ = self.A_fixed.copy()
        else:
            self.A_ = A

    @property
    def C(self):
        return self.C_

    @C.setter
    def C(self, C):
        if self.C_fixed is not None:
            self.C_ = self.C_fixed.copy()
        else:
            self.C_ = C
        if self.C_ is not None:
            self.C_inv = np.linalg.inv(self.C_)

    @property
    def pi(self):
        return self.pi_

    @pi.setter
    def pi(self, pi):
        if self.pi_fixed is not None:
            self.pi_ = self.pi_fixed.copy()
        else:
            self.pi_ = pi

    @property
    def W_fixed(self):
        return self.W_fixed_

    @W_fixed.setter
    def W_fixed(self, W):
        self.W_fixed_ = W
        self.W = W

    @property
    def A_fixed(self):
        return self.A_fixed_

    @A_fixed.setter
    def A_fixed(self, A):
        self.A_fixed_ = A
        self.A = A

    @property
    def C_fixed(self):
        return self.C_fixed_

    @C_fixed.setter
    def C_fixed(self, C):
        self.C_fixed_ = C
        self.C = C

    @property
    def pi_fixed(self):
        return self.pi_fixed_

    @pi_fixed.setter
    def pi_fixed(self, pi):
        self.pi_fixed_ = pi
        self.pi = pi

    @property
    def s_exps(self):
        return self.s_exps_

    @s_exps.setter
    def s_exps(self, s_exps):
        self.s_exps_ = s_exps

    @s_exps.getter
    def s_exps(self):
        assert self.X is not None
        return self.s_exps_

    @property
    def ss_exps(self):
        return self.ss_exps_

    @ss_exps.setter
    def ss_exps(self, ss_exps):
        self.ss_exps_ = ss_exps

    @ss_exps.getter
    def ss_exps(self):
        assert self.X is not None
        return self.ss_exps_

    @property
    def sstm1_exps(self):
        return self.sstm1_exps_

    @sstm1_exps.setter
    def sstm1_exps(self, sstm1_exps):
        self.sstm1_exps_ = sstm1_exps

    @sstm1_exps.getter
    def sstm1_exps(self):
        assert self.X is not None
        return self.sstm1_exps_

    @property
    def X(self):
        return self.X_

    @X.setter
    def X(self, X):
        self.X_ = X
        if X is not None:
            self._try_assign_parameters()
            self._initialize_expectations_none()


    def _try_assign_parameters(self):
        if self.W.shape[1] != self.X.shape[2]:
            raise ValueError("`o` dimension mismatch between W and X")
        self.n_samples = self.X.shape[0]
        self.T = self.X.shape[1]
        self.o = self.X.shape[2]

    def _initialize_expectations_none(self):
        self.s_exps = [None] * self.n_samples
        self.ss_exps = [None] * self.n_samples
        self.sstm1_exps = [None] * self.n_samples
        self.hessian_samples = [None] * self.n_samples

    def _gibbs(self, sample_idx):
        x = self.X[sample_idx]
        td = list(product(range(self.T), range(self.d)))
        for i in range(self.gibbs_max_iter):
            for t in range(self.T):
                old_s = self.s.copy()  # store prior to `d` updates
                for d in range(self.d):
                    sk = old_s.copy()
                    for k in range(self.k):
                        sk[t, d, :] = 0
                        sk[t, d, k] = 1

                        # Edge case -- end of sequence
                        if t == self.T-1:
                            A_tp1 = 1
                        else:
                            state_tp1_idx = np.argmax(sk[t+1, d, :])
                            A_tp1 = np.exp(self.A[d, state_tp1_idx, k])

                        # Edge case -- beginning of sequence
                        if t == 0:
                            A_tm1 = self.pi[d, k]
                        else:
                            state_tm1_idx = np.argmax(sk[t-1, d, :])
                            A_tm1 = np.exp(self.A[d, k, state_tm1_idx])

                        y_mu = np.einsum('dok,dk', self.W, sk[t, :, :])

                        pyt = norm_pdf_multivariate(x[t, :], y_mu, self.C)

                        self.ps[i, t, d, k] = A_tm1 * pyt * A_tp1 + self.zero_probability

                    # Normalize
                    self.ps[i, t, d, :] /= self.ps[i, t, d, :].sum(keepdims=True)

                    idx = np.random.choice(range(self.k), p=self.ps[i, t, d, :])

                    # Update s with this chain values
                    self.s[t, d, :] = 0
                    self.s[t, d, idx] = 1

            self.states[i] = self.s.copy()

        # Hidden state expectation estimation

        # \langle s^{t}_{i} \rangle_{\rm c}
        s_exp = np.zeros(shape=(self.T, self.d, self.k))
        for i in range(len(self.states)):
            s_exp += self.states[i]
        s_exp /= len(self.states)

        # \langle s^{t}_{i} (s^{t}_{j})^{\rm T} \rangle_{\rm c}
        ss_exp = np.zeros(shape=(self.T, self.d, self.d, self.k, self.k))
        for i in range(len(self.states)):
            for t, d1 in td:
                for d2 in range(self.d):
                    ss_exp[t, d1, d2] += np.outer(self.states[i, t, d1, :],
                                                  self.states[i, t, d2, :])

        ss_exp /= len(self.states)

        # \langle s^{t}_{i} (s^{t-1}_{i})^{\rm T} \rangle_{\rm c}
        sstm1_exp = np.zeros(shape=(self.T, self.d, self.k, self.k))
        for i in range(len(self.states)):
            for t, d in td:
                if t == 0:
                    continue
                sstm1_exp[t, d] += np.outer(self.states[i, t, d, :],
                                            self.states[i, t-1, d, :])
        sstm1_exp /= len(self.states)

        return s_exp, ss_exp, sstm1_exp

    def _mean_field(self, sample_idx, randomize=False):
        x = self.X[sample_idx]

        kld_best = np.inf
        s_exp_best = None
        ss_exp_best = None
        sstm1_exp_best = None
        for jt in range(self.e_step_retries+1):
            # if jt == 0:
            #     if (len(self.s_exps) < self.n_samples) or (randomize):
            #         m = np.random.uniform(size=(self.T*self.d*self.k)).reshape(self.T, self.d, self.k)
            #         for t in range(self.T):
            #             for d in range(self.d):
            #                 m[t, d, :] = softmax(m[t, d, :])
            #     else:
            #         m = self.s_exps[sample_idx].copy()
            # else:
            m = np.random.uniform(size=(self.T*self.d*self.k)).reshape(self.T, self.d, self.k)
            for t in range(self.T):
                for d in range(self.d):
                    m[t, d, :] = softmax(m[t, d, :])


            kld_old = np.inf

            for it in range(self.mean_field_max_iter):
                td = list(product(range(self.T), range(self.d)))
                for t, d in sorted(td, key=lambda x: np.random.random()):
                    wm = np.einsum('dok,dk', self.W, m[t])
                    y_err = x[t] - wm

                    # Update and normalize the vector m[t, d, :]
                    log_m_new = np.zeros(shape=(self.k,))
                    for k in range(self.k):
                        if t == 0:
                            am = np.log(self.pi[d, k])
                        else:
                            am = self.A[d, k, :].dot(m[t-1, d, :])

                        if t == self.T-1:
                            ma = 0
                        else:
                            ma = m[t+1, d, :].dot(self.A[d, :, k])

                        log_m_new[k] = self.W[d, :, k].dot(self.C_inv.dot(y_err)) \
                                     + self.W[d, :, k].dot(self.C_inv.dot(self.W[d].dot(m[t, d, :])))\
                                     - 1/2 * self.W[d, :, k].dot(self.C_inv.dot(self.W[d, :, k])) \
                                     - 1 + ma + am
                    #m[t, d, :] = softmax(log_m_new) + self.zero_probability
                    m[t, d, :] = np.clip(softmax(log_m_new), self.zero_probability, 1-self.zero_probability)
                    m[t, d, :] /= m[t, d, :].sum()

                # Hidden state expectation estimation
                s_exp = m

                mm = np.zeros(shape=(self.T, self.d, self.k, self.d, self.k))
                for t in range(self.T):
                    mm[t] = np.outer(m[t].ravel(), m[t].ravel()).reshape(self.d,
                                                                         self.k,
                                                                         self.d,
                                                                         self.k)

                # Fix diagonal d1 = d2 case:
                for t in range(self.T):
                    for d in range(self.d):
                        mm[t, d, :, d, :] = np.diag(m[t, d, :])

                ss_exp = np.swapaxes(mm, 2, 3)

                sstm1_exp = np.zeros(shape=(self.T,
                                            self.d,
                                            self.k,
                                            self.k))

                for t in range(1, self.T):
                    for d in range(self.d):
                        sstm1_exp[t, d, :, :] = np.outer(m[t, d, :], m[t-1, d, :])

                kld = self.kld_sample_mean_field(sample_idx, s_exp, ss_exp, sstm1_exp)

                if np.abs(kld-kld_old) < self.mean_field_kld_tol:
                    break
                else:
                    kld_old = kld

            if kld <= kld_best:
                kld_best = kld
                s_exp_best = s_exp
                ss_exp_best = ss_exp
                sstm1_exp_best = sstm1_exp

        if (it == self.mean_field_max_iter-1) and (self.verbose):
            warnings.warn("== Reached mean_field_max_iter")
            print('Reached mean_field_max_iter!')

        return s_exp_best, ss_exp_best, sstm1_exp_best

    def forward_backward(self, h):
        # eps = np.finfo(np.float32).eps
        eps = 0

        # Alpha -- "forward"
        alpha = np.zeros(shape=(self.T, self.d, self.k))
        alpha[0] = self.pi * h[0]
        for t in range(1, self.T):
            alpha[t] = np.einsum('dkj,dj,dk->dk', np.exp(self.A), alpha[t-1], h[t])
            alpha[t] /= alpha[t].sum(axis=1, keepdims=True)

        # Beta -- "backward"
        beta = np.zeros(shape=(self.T, self.d, self.k))
        beta[self.T-1, :, :] = 1
        for t in range(self.T-2, -1, -1):
            beta[t] = np.einsum('djk,dj,dj->dk', np.exp(self.A), h[t+1], beta[t+1])
            beta[t] /= beta[t].sum(axis=1, keepdims=True)

        # alpha += eps
        # beta += eps

        # Gamma -- expectation
        gamma = alpha * beta / (np.sum(alpha * beta, axis=2, keepdims=True))

        # Expectation values:
        s_exp = gamma.copy()

        ss_exp = np.einsum('tdk,tbl->tdbkl', gamma, gamma)
        # Fix up the diagonal case (d1 == d2)
        for t in range(self.T):
            for d in range(self.d):
                ss_exp[t, d, d] = np.diag(gamma[t, d])

        sstm1_exp = np.zeros(shape=(self.T, self.d, self.k, self.k))
        for d in range(self.d):
            for t in range(1, self.T):
                norm = np.einsum('k,lk,l,l', alpha[t-1, d], np.exp(self.A[d]), h[t, d], beta[t, d])
                sstm1_exp[t, d] = np.einsum('k,lk,l,l->kl', alpha[t-1, d], np.exp(self.A[d]), h[t, d], beta[t, d]) / norm

        return s_exp, ss_exp, sstm1_exp

    def _sva(self, sample_idx, randomize=False):
        x = self.X[sample_idx]

        # First time through -- initialize probability matrix for all samples
        if self.h is None:
            self.h = np.random.uniform(self.zero_probability, 1 - self.zero_probability,
                                       size=(self.n_samples*self.T*self.d*self.k))\
                              .reshape(self.n_samples, self.T, self.d, self.k)
            for t in range(self.T):
                for d in range(self.d):
                    self.h[sample_idx, t, d, :] = softmax(self.h[sample_idx, t, d, :])

        h = self.h[sample_idx].copy()

        kld_best = np.inf
        s_exp_best = None
        ss_exp_best = None
        sstm1_exp_best = None
        h_best = None
        change = 0
        for jt in range(self.e_step_retries+1):
            # if jt == 0:
            #     # If we don't have an estimate of all the samples yet, continue initializing
            #     # Else load previous best estimate, for updating based on new WACpi.
            #     # if len(self.s_exps) < self.n_samples:
            #     #     s_exp, ss_exp, sstm1_exp = self.forward_backward(h)
            #     # else:
            #     s_exp, ss_exp, sstm1_exp = self.s_exps[sample_idx], self.ss_exps[sample_idx], self.sstm1_exps[sample_idx]
            # else:
            #     # Random retries to hop into better local/global optima
            #     h = np.random.uniform(self.zero_probability, 1 - self.zero_probability,
            #                           size=(self.T*self.d*self.k))\
            #                  .reshape(self.T, self.d, self.k)
            #     for t in range(self.T):
            #         for d in range(self.d):
            #             h[t, d, :] = softmax(h[t, d, :])
            s_exp, ss_exp, sstm1_exp = self.forward_backward(h)

            kld_old = np.inf
            for it in range(self.sva_max_iter):
                td = list(product(range(self.T), range(self.d)))
                for t, d in sorted(td, key=lambda x: np.random.random()):
                    y_err = np.zeros(shape=(self.o,))
                    ws = 0
                    for dm in range(self.d):
                        if dm == d:
                            continue
                        ws += self.W[dm].dot(s_exp[t, dm])
                    y_err = x[t] - ws

                    # Update and normalize the vector h[t, d, :]
                    log_h_new = np.einsum('ok,o->k', self.W[d], self.C_inv.dot(y_err)) \
                                - 1/2 * np.einsum('ok,op,pk->k', self.W[d], self.C_inv, self.W[d])
                    #h[t, d, :] = softmax(log_h_new) + self.zero_probability
                    h[t, d, :] = np.clip(softmax(log_h_new), self.zero_probability, 1-self.zero_probability)
                    h[t, d, :] /= h[t, d, :].sum()

                s_exp, ss_exp, sstm1_exp = self.forward_backward(h)
                self.h[sample_idx] = h.copy()  # Required to store this for KLD computation

                kld = self.kld_sample_sva(sample_idx, s_exp, ss_exp, sstm1_exp)

                if np.abs(kld-kld_old) < self.sva_kld_tol:
                    break
                else:
                    kld_old = kld

            if kld < kld_best:
                kld_best = kld
                s_exp_best = s_exp.copy()
                ss_exp_best = ss_exp.copy()
                sstm1_exp_best = sstm1_exp.copy()
                h_best = h.copy()
                change += 1

        if (it == self.sva_max_iter-1) and (self.verbose):
            warnings.warn("== Reached sva_max_iter")
            print('Reached sva_max_iter!')

        # Don't forget to store the best for the next time through the E-step!
        if h_best is None:
            self.h[sample_idx] = h.copy()
            return s_exp, ss_exp, sstm1_exp
        else:
            self.h[sample_idx] = h_best.copy()
            return s_exp_best, ss_exp_best, sstm1_exp_best

    def py(self, sample_idx):
        """ Probability of observable given parameters and hidden states, for
        all hidden state realizations.

        Parameters
        ----------

        Returns
        -------
        py: np.array of shape (T, k**d)
            First index is time second is realization (or hidden state
            configuration).
        """
        x = self.X[sample_idx]
        realizations = self.realizations()


        py = jit_py(self.T, self.d, self.k, self.o, x, realizations, self.W, self.C)
        return py

    @lru_cache(maxsize=None)
    def realizations(self):
        """ A representation of all hidden state realizations. The entries
        are the index corresponding to the state.  For example, for a system
        with three level hidden states the evaluation realizations[1, 3]
        yielding a value of 2 means chain index 1 in realization index 3
        implies chain 1's hidden state is [0, 0, 1], when represented as
        a binary vector.

        Returns
        -------

        realizations: np.array of shape (d, k**d)
        """
        index_states = [list(range(self.k)) for d in range(self.d)]
        realizations = np.array(list(product(*index_states))).T
        return realizations

    @staticmethod
    @lru_cache(maxsize=None)
    def get_k_contrib(d, k, string_realizations):
        realizations = np.frombuffer(string_realizations, dtype=int).reshape(d, -1)
        k_contrib = defaultdict(set)  # mapping (t, d, k) -> i indices of realization containing k
        for idx_d in range(d):
            for i in range(realizations.shape[1]):
                k_contrib[(idx_d, realizations[idx_d, i])].add(i)

        k_contrib_array = np.zeros(shape=(d, k, k**(d-1)), dtype=int)

        for (i, j), el_set in k_contrib.items():
            k_contrib_array[i, j, :] = np.array(list(el_set))

        return k_contrib_array

    def _exact(self, sample_idx):
        realizations = self.realizations()
        py = self.py(sample_idx)


        k_contrib = self.get_k_contrib(self.d, self.k, realizations.tobytes())
        s_exp, ss_exp, sstm1_exp = jit_exact(self.T,
                                                self.d,
                                                self.k,
                                                realizations,
                                                k_contrib,
                                                py,
                                                self.A,
                                                self.pi)
        return s_exp, ss_exp, sstm1_exp

    def kld_sample_mean_field(self, sample_idx, s_exp, ss_exp, sstm1_exp):
        # eps = np.finfo(np.float32).eps
        eps = 0
        exp_log_Ptilde = 0

        log_Z = 0  #self.T*self.o/2 * np.log(2*np.pi) - self.T/2 * np.log(np.linalg.det(self.C_inv))

        for t in range(self.T):
            exp_log_Ptilde += np.einsum('ij,ij',
                                        s_exp[t, :, :],
                                        np.log(s_exp[t, :, :] + eps))
        exp_log_Ptilde -= log_Z

        exp_log_P = self.expected_complete_log_likelihood_sample(sample_idx,
                                                                 s_exp,
                                                                 ss_exp,
                                                                 sstm1_exp)

        kld = exp_log_Ptilde - exp_log_P

        return kld

    def kld_sample_sva(self, sample_idx, s_exp, ss_exp, sstm1_exp):
        # eps = np.finfo(np.float32).eps
        eps = 0
        exp_log_Ptilde = 0

        log_Z = 0
        for d in range(self.d):
            log_Z += np.log(self.h[sample_idx, 0, d, :].dot(self.pi[d, :]))

        Ass = np.sum(np.einsum('djk,tdjk->t', self.A, sstm1_exp)[1:])

        slogpi = np.einsum('dk, dk', s_exp[0], np.log(self.pi))

        for t in range(self.T):
            exp_log_Ptilde += np.einsum('ij,ij',
                                        s_exp[t, :, :],
                                        np.log(self.h[sample_idx, t, :, :] + eps))
        exp_log_Ptilde += Ass + slogpi - log_Z

        exp_log_P = self.expected_complete_log_likelihood_sample(sample_idx,
                                                                 s_exp,
                                                                 ss_exp,
                                                                 sstm1_exp)

        kld = exp_log_Ptilde - exp_log_P

        return kld

    def kld(self):
        """ Returns the Kullback-Leibler Divergence for SVA and Mean
        Field methods.

        Returns
        -------
        kld: float
        """
        if len(self.s_exps) < self.n_samples:
            return 1E10

        kld = 0
        for sample_idx in range(len(self.X)):
            if self.method == 'mean_field':
                kld += self.kld_sample_mean_field(sample_idx,
                                                  self.s_exps[sample_idx],
                                                  self.ss_exps[sample_idx],
                                                  self.sstm1_exps[sample_idx])
            elif self.method == 'sva':
                kld += self.kld_sample_sva(sample_idx,
                                           self.s_exps[sample_idx],
                                           self.ss_exps[sample_idx],
                                           self.sstm1_exps[sample_idx])

        return kld

    def expected_complete_log_likelihood_sample(self, sample_idx, s_exp, ss_exp, sstm1_exp):
        """ The expected complete log likelihood for a particular sample.
        """
        y = self.X[sample_idx]

        yCy = np.einsum('to,op,tp', y, self.C_inv, y)

        yCWs = np.einsum('to,op,dpk,tdk', y, self.C_inv, self.W, s_exp)

        WCWss = np.sum(np.einsum('dok,op,epl,tdekl', self.W, self.C_inv, self.W, ss_exp))

        Ass = np.sum(np.einsum('djk,tdjk->t', self.A, sstm1_exp)[1:])

        slogpi = np.einsum('dk, dk', s_exp[0], np.log(self.pi))

        log_Z = self.T*self.o/2 * np.log(2*np.pi) - self.T/2 * np.log(np.linalg.det(self.C_inv)) \
                + self.d * (self.T-1) * np.log(self.k)

        exp_log_P = -1/2 * yCy + yCWs - 1/2 * WCWss + Ass + slogpi - log_Z
        return exp_log_P

    def expected_complete_log_likelihood(self, s_exps=None, ss_exps=None, sstm1_exps=None):
        r""" The expected complete log likelihood for all samples.

        Returns
        -------
        exp_log_P: float
            expected complete log likelihood

        Notes
        -----
        .. math:: \langle \log P(Y, S| \phi) \rangle
        """
        if len(self.s_exps) < self.n_samples:
            return -np.inf

        if s_exps is None:
            s_exps = self.s_exps

        if ss_exps is None:
            ss_exps = self.ss_exps

        if sstm1_exps is None:
            sstm1_exps = self.sstm1_exps

        exp_log_P = 0
        for sample_idx in range(self.n_samples):
            exp_log_P += self.expected_complete_log_likelihood_sample(sample_idx,
                                                                      s_exps[sample_idx],
                                                                      ss_exps[sample_idx],
                                                                      sstm1_exps[sample_idx])
        return exp_log_P

    def log_likelihood_sample(self, sample_idx):
        r""" The log likelihood for a particular sample.

        Parameters
        ----------
        sample_idx: int
            The example index (corresponding data selection X[sample_idx, ...])

        Returns
        -------
        ll: float
            Sample log likelihood.

        Notes
        -----
        Complexity :math:`\mathcal{O}(Tdk^{d+1})`.
        """
        realizations = self.realizations()
        py = self.py(sample_idx)

        # Normalizations from the forward pass for calculating alpha

        c, _ = jit_forward(self.T,
                            self.d,
                            self.k,
                            realizations,
                            py,
                            self.A,
                            self.pi)

        ll = np.sum(np.log(c))

        return ll

    def log_likelihood(self):
        r""" Log likelihood of the entire data given the model parameters.

        Returns
        -------
        ll: float, (R+)
            Log likelihood.

        Notes
        -----
        Complexity :math:`\mathcal{O}(NTdk^{d+1})`.
        """
        ll = 0
        for sample_idx in range(len(self.X)):
            ll += self.log_likelihood_sample(sample_idx)

        return ll

    def P(self, sample_idx, s):
        """Simply evaluating the probability function

        Parameters
        ----------
        sample_idx : int
            The sample index for which to compute P.
        s : np.array
            The actual hidden states trajectories.
        """
        y = self.X[sample_idx]

        yCy = np.einsum('to,op,tp', y, self.C_inv, y)

        yCWs = np.einsum('to,op,dpk,tdk', y, self.C_inv, self.W, s)

        WsCWs = np.einsum('dok,tdk,op,epl,tel', self.W, s, self.C_inv, self.W, s)

        sAs = 0
        for t in range(1, self.T):
            sAs += np.einsum('dk,dkj,dj', s[t], self.A, s[t-1])

        slogpi = np.einsum('dk, dk', s[0], np.log(self.pi))

        log_Z = self.T*self.o/2 * np.log(2*np.pi) - self.T/2 \
                * np.log(np.linalg.det(self.C_inv)) \
                + self.d * (self.T-1) * np.log(self.k)

        return 1/np.exp(log_Z) * np.exp(-1/2 * yCy + yCWs - 1/2 * WsCWs + sAs + slogpi)

    def update_W(self):
        if self.W_fixed is not None:
            return self.W_fixed
        else:
            return self.estimate_W()

    def estimate_W(self):
        # W update
        sy = np.zeros(shape=(self.d*self.k, self.o))

        for n in range(len(self.s_exps)):
            for t in range(self.T):
                sy += np.outer(self.s_exps[n][t].ravel(), self.X[n, t].ravel())

        # Sum over samples
        ss = sum(self.ss_exps)  # shape = (t, d, d, k, k)
        # Sum over time
        ss = ss.sum(axis=0)  # shape = (d, d, k, k)
        ss = np.swapaxes(ss, 1, 2).reshape(self.d*self.k, self.d*self.k)
        random_adj = 1E-10*np.random.rand(self.d*self.k*self.d*self.k)\
                                    .reshape(self.d*self.k, self.d*self.k)
        ss += self.n_samples * self.T * random_adj

        try:
            ss_exp_inv = np.linalg.pinv(ss)
        except LinAlgError as e:
            return self.W

        W = np.swapaxes(ss_exp_inv.dot(sy).reshape(self.d, self.k, self.o), 1, 2)

        if self.stochastic_training:
            W += self.stochastic_lr**(np.log(self.it+2)) * 10 * np.random.rand(*W.shape)
        W = self.canonically_transform_W(W)

        return W

    @staticmethod
    def canonically_transform_W(W_in):
        """ Transforms W such that the overall mean is shifted to the
        W[0, ...] element, and the W[1:d, ...] elements have zero mean
        along axis -1. The Hessian calculation expects, and model
        training enforces, canonical W.

        Parameters
        ----------
        W_in: np.array of shape (d, o, k)
            Initial W matrix.

        Returns
        -------
        W: np.array of shape (d, o, k)
            Canonically transformed W matrix.
        """
        W = W_in.copy()
        d, o, k = W.shape
        for idx_o in range(o):
            d_means = []
            for idx_d in range(1, d):
                d_means.append(np.mean(W[idx_d, idx_o, :]))

            W[0, idx_o, :] += np.sum(d_means)
            for idx_d in range(1, d):
                W[idx_d, idx_o, :] -= d_means[idx_d-1]
        return W

    def update_A(self):
        if self.A_fixed is not None:
            return self.A_fixed
        else:
            return self.estimate_A()

    def estimate_A(self):
        # A update
        sstm1 = sum(self.sstm1_exps)  # Sum over samples
        A = np.sum(sstm1, axis=0) + self.zero_probability  # Sum over time

        if self.stochastic_training:
            A += self.stochastic_lr**(np.log(self.it+2)) * 10 * np.random.rand(*A.shape)
        A = np.log(A) - np.log(np.sum(A, axis=1, keepdims=True))

        return A

    def update_C(self):
        if self.C_fixed is not None:
            return self.C_fixed
        else:
            return self.estimate_C()

    def estimate_C(self):
        W = self.W.copy()
        C = 0
        for n in range(self.n_samples):
            for t in range(self.T):
                y = self.X[n, t]
                Ws = np.einsum('dok,dk', W, self.s_exps[n][t])
                Wss = np.einsum('dok,idjk->oij', W, self.ss_exps[n][t])
                WWss = np.einsum('dnk,odk->no', W, Wss)

                yy = np.outer(y, y)
                yWs = np.outer(y, Ws)
                Wsy = np.outer(Ws, y)

                C += yy - yWs - Wsy + WWss

        C /= (self.T*self.n_samples)

        return C

    def update_pi(self):
        if self.pi_fixed is not None:
            return self.pi_fixed
        else:
            return self.estimate_pi()

    def estimate_pi(self):
        pi = 0
        for n in range(self.n_samples):
            pi += self.s_exps[n][0]  # t=0, shape=(d, k)
        pi += self.zero_probability
        if self.stochastic_training:
            pi += self.stochastic_lr**(np.log(self.it+2)) * 10 * np.random.rand(*pi.shape)
        pi /= pi.sum(axis=1, keepdims=True)

        return pi


    def hessian_for_sample(self, sample_idx):
        r""" Calculates the Hessian of the log likelihood for a particular sample.
        Expects canonically transformed W.  Canonical W is enforced during model
        training.

        Parameters
        ----------
        sample_idx: int
            Index of self.X to compute hessian over

        Returns
        -------
        hessian_: np.array of shape (Np, Np) [see Notes below]
            Square array where the rows and columns are ordered by
            the independent elements of W, A, C, and pi, in that order.

        Notes
        -----
        The number of independent W elements is :math:`dok-(d-1)o`; for A
        this is :math:`d(k-1)k`; for C this is :math:`o^{2}`; and for pi
        this is :math:`d(k-1)`. So,

        .. math:: N_p = dok - (d-1)o + d(k-1)k + o^{2} + d(k-1)
        """
        if self.hessian_samples[sample_idx] is not None:
            return self.hessian_samples[sample_idx]

        realizations = self.realizations()
        k_contrib = self.get_k_contrib(self.d, self.k, realizations.tobytes())
        x = self.X[sample_idx, :, :].copy()
        py = self.py(sample_idx)

        hessian = Hessian(self.T, self.d, self.o, self.k, x, self.W, self.A,
                          self.C, self.pi, self.C_inv, realizations, 
                          k_contrib, py)


        self.hessian_samples[sample_idx] = hessian.hessian()

        return self.hessian_samples[sample_idx]

    def hessian(self):
        r""" Calculates the Hessian of the log likelihood averaging over all samples.
        Expects canonically transformed W.  Canonical W is enforced during model
        training.

        Returns
        -------
        hessian_: np.array of shape (Np, Np) [see Notes below]
            Square array where the rows and columns are ordered by
            the independent elements of W, A, C, and pi, in that order.

        Notes
        -----
        The number of independent W elements is :math:`dok-(d-1)o`; for A
        this is :math:`d(k-1)k`; for C this is :math:`o^{2}`; and for pi
        this is :math:`d(k-1)`. So,

        .. math:: N_p = dok - (d-1)o + d(k-1)k + o^{2} + d(k-1)
        """

        if self.hessian_ is not None:
            return self.hessian_

        all_hessians = [self.hessian_for_sample(sample_idx) for sample_idx in range(self.n_samples)]

        self.hessian_ = np.mean(all_hessians, axis=0)
        return self.hessian_

    def standard_errors(self):
        # TODO: Average across all hessians
        r""" Calculates the standard error for each parameter based on the
        Hessian of the log likelihood.

        Notes
        -----
        With :math:`I = -H`, the negative of the Hessian, the
        observed information matrix, the standard errors are estimated
        as follows:

        .. math:: d\theta_{i} = \sqrt{(I^{-1})_{ii}}
        """
        oim = -self.hessian()

        endW = np.prod(self.W.shape)
        endA = endW + np.prod(self.A.shape)
        endC = endA + np.prod(self.C.shape)
        endpi = endC + np.prod(self.pi.shape)

        endWr = np.prod(self.W.shape) - (self.d-1)*self.o
        endAr = endWr + np.prod(self.A[:, :-1, :].shape)
        endCr = endAr + np.prod(self.C.shape)
        endpir = endCr + np.prod(self.pi[:, :-1].shape)

        # Estimate confidence intervals from Observed Information Matrix
        confidence_intervals = np.sqrt(np.linalg.inv(oim).diagonal())

        self.dC = confidence_intervals[endAr:endCr].reshape(self.C.shape)

        # Add back in the dependent variable variances
        # NOTE: This depends on the 'canonically' transformed W
        dW_red = confidence_intervals[:endWr]
        dW_first_d = dW_red[:1*self.o*self.k].reshape(1, self.o, self.k)
        dW_remaining_d = dW_red[1*self.o*self.k:].reshape(self.d-1, self.o, self.k-1)
        dW_dep = np.sqrt(np.einsum('dok->do', dW_remaining_d**2)[:, :, np.newaxis])
        dW_remaining_d = np.concatenate([dW_remaining_d, dW_dep], axis=2)
        self.dW = np.concatenate([dW_first_d, dW_remaining_d], axis=0)

        dA_red = confidence_intervals[endWr:endAr].reshape(self.d, self.k-1, self.k)
        A_dep_prob = (1 - np.einsum('djk->dk', np.exp(self.A[:, :-1, :])))
        dA_dep = np.sqrt(np.einsum('djk,djk->dk',
                                   dA_red**2,
                                   np.exp(2 * self.A[:, :-1, :])) / A_dep_prob**2)
        dA_dep = dA_dep[:, np.newaxis, :]
        self.dA = np.concatenate([dA_red, dA_dep], axis=1)

        dpi_red = confidence_intervals[endCr:endpir].reshape(self.d, self.k-1)
        dpi_dep = np.sqrt(np.einsum('dj->d', dpi_red**2)[:, np.newaxis])
        self.dpi = np.concatenate([dpi_red, dpi_dep], axis=1)

        return self.dW, self.dA, self.dC, self.dpi


    def E(self, X, **kwargs):
        """ Expectation step of the EM algorithm.  Updates the
        state expectations.

        Parameters
        ----------
        X: np.array of shape (N, T, o)
        kwargs: arguments
            keyword arguments to be passed to whichever method is being used.

        Returns
        -------
        s_exp, ss_exp, sstm1_exp: tuple
            State expectations.
        """
        if self.method == 'gibbs':
            return self._gibbs(X, **kwargs)
        elif self.method == 'mean_field':
            return self._mean_field(X, **kwargs)
        elif self.method == 'sva':
            return self._sva(X, **kwargs)
        elif self.method == 'exact':
            return self._exact(X, **kwargs)

    def M(self):
        """ Parameter Maximization step of the EM algorithm.

        Returns
        -------
        W, A, C, pi: tuple
            Model parameters.
        """
        W = self.update_W()
        A = self.update_A()
        C = self.update_C()
        pi = self.update_pi()

        return W, A, C, pi

    @property
    def is_fixed(self):
        return (self.W_fixed is not None) \
                and (self.A_fixed is not None) \
                and (self.C_fixed is not None) \
                and (self.pi_fixed is not None)

    def fix_fit_params(self):
        self.W_fixed = self.W
        self.A_fixed = self.A
        self.C_fixed = self.C
        self.pi_fixed = self.pi

    def unfix_fit_params(self):
        self.W_fixed_ = None
        self.A_fixed_ = None
        self.C_fixed_ = None
        self.pi_fixed_ = None

    def final_state_distribution(self):
        r""" Estimates the final hidden state distribution given the
        time series.  Uses the Viterbi algorithm to estimate the
        final time sample's hidden state.  Then uses the transition
        matrices to calculate the probability distribution for the
        next hidden state values.  Can be used as the initial hidden
        state distribution when splitting up the time series.

        Returns
        -------
        fs_probs: list of length N.
            A list of the next time sample distribution for each time series
            example.

        Notes
        -----
        Complexity :math:`\mathcal{O}(NTdk^{d+1})`.
        """
        fs_probs = []
        for sample_idx in range(self.X.shape[0]):
            vstates_train = self.viterbi(sample_idx)
            fs_probs.append(np.einsum('dkp,dp->dk', np.exp(self.A), vstates_train[-1]))

        # Average state occupancy probablity over all samples
        fs_probs = np.mean(np.array(fs_probs), axis=0)

        return fs_probs

    def _fit_single_em_iteration(self):
        self._update_expectations()
        self._try_print_pre_mstep_metrics()
        self.W, self.A, self.C, self.pi = self.M()
        self._try_print_post_mstep_metrics()

        self._save_to_trace_if_have_directory()

    def _update_expectations(self):
        for sample_idx in range(self.n_samples):
            s_exp, ss_exp, sstm1_exp = self.E(sample_idx)
            self.s_exps[sample_idx] = s_exp.copy()
            self.ss_exps[sample_idx] = ss_exp.copy()
            self.sstm1_exps[sample_idx] = sstm1_exp.copy()

    def _try_print_pre_mstep_metrics(self):
        if self.verbose:
            msg = self._sep_str() + '\n'\
                    + self._pre_mstep_str() \
                    + self._cllik_str() \
                    + self._estep_metric_str()
            print(msg)

    def _try_print_post_mstep_metrics(self):
        if self.verbose:
            msg = self._post_mstep_str() \
                    + self._cllik_str() \
                    + self._estep_metric_str()
            print(msg)

    def _sep_str(self):
        return '=' * 50

    def _pre_mstep_str(self):
        return "PRE -M-STEP:: "

    def _post_mstep_str(self):
        return "POST-M-STEP:: "

    def _cllik_str(self):
        cllik = self.expected_complete_log_likelihood()
        return "Complete Log Likelihood: {0:0.2f}".format(cllik)

    def _estep_metric_str(self):
        msg = ""
        if self.method in ('mean_field', 'sva'):
            msg = "\t KLD: {0:0.2f}".format(self.kld())
        elif self.method == 'exact':
            msg = "\t Log likelihood: {0:0.2f}".format(self.log_likelihood())
        return msg

    def _create_run_directory_and_save_x_if_needed(self):
        if self.save_directory is None:
            return

        if not hasattr(self, "_run_directory"):
            self._run_directory = self._build_run_directory()
        if not os.path.exists(self._run_directory):
            os.makedirs(self._run_directory)

        if self.save_x:
            self._save_x_to_file()

    def _save_x_to_file(self):
        self._x_directory = self._build_snapshot_filepath_for_x()
        with open(self._x_directory, 'wb') as f:
            pickle.dump(self.X, f)
    
    def _save_to_trace_if_have_directory(self):
        if self.save_directory is None:
            return

        self._fit_trace_directory = self._build_fit_trace_directory()

        if not os.path.exists(self._fit_trace_directory):
            os.makedirs(self._fit_trace_directory)

        fit_trace_data = self._build_snapshot_data()
        ftd_filepath = self._build_snapshot_filepath()

        with open(ftd_filepath, 'wb') as f:
            pickle.dump(fit_trace_data, f)

    @classmethod
    def load_model_from_file(cls, filename, load_x_from_snapshot=False, x_filepath=None):
        """ Generate a FHMM instance from a saved pickle file

        Parameters
        ----------
        filename: str
            The pkl file with data to load
        load_x_from_snapshot: bool, False
            Load saved X specified in snapshot during fitting, if it exists
        x_filepath : str, optional
            Specify a file location if you want to load X from a file (that is different than what is saved during fitting).
            If specified, this will ignore the load_x_from_snapshot keyword argument

        Returns
        -------
        new_fhmm: FHMM
            FHMM instance generated from saved data

        """
        # Open files and read in data
        with open(filename,"rb") as f:
            fit_data = pickle.load(f)

        # Create new instance and pass in data
        new_fhmm = cls()
        new_fhmm.set_params(**fit_data['params'])
        new_fhmm.W = fit_data['W']
        new_fhmm.A = fit_data['A']
        new_fhmm.C = fit_data['C']
        new_fhmm.pi = fit_data['pi']
        # new_fhmm.X = new_X

        if load_x_from_snapshot and not x_filepath:
            x_filepath = fit_data["x_filepath"]
            assert x_filepath is not None, "No file location for X was defined"

        if x_filepath:
            with open(x_filepath,"rb") as f:
                new_X = pickle.load(f)
            new_fhmm.X = new_X
            new_fhmm._update_expectations()

        return new_fhmm

    @classmethod
    def load_best_models_from_directories(cls, directories, load_x_from_snapshot=False, x_filepath=None):
        """ Loads best restart in the given directories. Assumes directories is set up from a previous FHMM fitting run.

        Parameters
        ----------
        directories : _type_
            _description_
        load_x_from_snapshot : bool, optional
            _description_, by default False
        x_filepath : _type_, optional
            _description_, by default None
        """
        model_list = [cls.load_final_model_from_data(directory, load_x_from_snapshot=load_x_from_snapshot, x_filepath=x_filepath) for directory in directories]
        return model_list

    @classmethod
    def load_final_model_from_data(cls, save_directory, load_x_from_snapshot=False, x_filepath=None):
        best_iteration_path = None
        best_score = -np.inf
        for restart_directory in sorted(os.listdir(save_directory)):
            if not os.path.isfile(os.path.join(save_directory, restart_directory)):
                files = os.listdir(os.path.join(save_directory, restart_directory))
                idx_sorted = np.argsort([int(f[2:-4]) for f in files])
                final_iteration = files[idx_sorted[-1]]
                final_iteration_path = os.path.join(save_directory, restart_directory, final_iteration)
                with open(final_iteration_path, 'rb') as f:
                    fit_data = pickle.load(f)
                if fit_data['score'] > best_score:
                    best_score = fit_data['score']
                    best_iteration_path = final_iteration_path
        return FHMM.load_model_from_file(best_iteration_path, load_x_from_snapshot=load_x_from_snapshot, x_filepath=x_filepath)

    def _build_snapshot_data(self):
        fit_data = {'id': id(self),
                    'it': self.it,
                    'params': self.get_params(),
                    'W': self.W,
                    'A': self.A,
                    'C': self.C,
                    'pi': self.pi,
                    'score': self.score(),
                    'x_filepath' : self._x_directory if self.save_x else None,
                   }

        return fit_data

    def _build_run_directory(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return os.path.join(self.save_directory, timestamp)

    def _build_fit_trace_directory(self):
        return os.path.join(self._run_directory, 'id' + str(id(self)))

    def _build_snapshot_filepath(self):
        return os.path.join(self._fit_trace_directory, 'it' + str(self.it) + '.pkl')

    def _build_snapshot_filepath_for_x(self):
        return os.path.join(self._run_directory, 'X' + '.pkl')

    def _fit_em_iterations(self):
        for it in range(self.em_max_iter):
            self.it = it
            self._fit_single_em_iteration()
            if self.convergence_monitor.update_has_converged():
                break
        self.convergence_monitor.reset()
        if (self.it == self.em_max_iter - 1) and self.verbose:
            print("Reached em_max_iter -- EXITING..")

    def fit(self, X, y=None):
        """ Fit the FHMM model to data X using an EM algorithm.

        Parameters
        ----------
        X: np.array of shape (N, T, o)
            The input observable data.
        y: np.array, None
            Labels for the examples; unused.
        """
        self.X = X
        if self._is_fixed_warn(): return self
        self._create_run_directory_and_save_x_if_needed()
        # results = self.sequential_fit()
        results = self.parallel_fit()
        self._fix_self_to_best_model(results)

        return self

    def _is_fixed_warn(self):
        if self.is_fixed:
            warnings.warn("Model fixed! Exiting...")
        return self.is_fixed

    def sequential_fit(self):
        results = []
        for n_start in range(self.n_restarts+1):
            results.append(self._n_restart_fit(self))
        return results

    def parallel_fit(self):
        futures = []
        for n_start in range(self.n_restarts+1):
            futures.append(delayed(self._n_restart_fit)(self))
        results = Parallel()(futures)
        return results

    def score(self):
        if self.method == 'exact':
            return self.log_likelihood()
        elif self.method in ('mean_field', 'sva'):
            return -self.kld()
        else:
            return self.expected_complete_log_likelihood()

    @staticmethod
    def _n_restart_fit(self):
        fhmm = deepcopy(self)
        fhmm.initialize_parameters()
        fhmm._fit_em_iterations()
        fhmm.fix_fit_params()
        result = (fhmm.score(), fhmm.get_params())
        return result

    def _fix_self_to_best_model(self, results):
        self.results_ = sorted(results, key=lambda x: x[0], reverse=True)
        best_params = self.results_[0][1]
        self.set_params(**best_params)
        self._update_expectations()

    def viterbi(self, sample_idx):
        r""" The Viterbi algorithm for FHMMs.

        Parameters
        ----------
        sample_idx: int
            The time series example index on which to apply algorithm,
            chosen via X[sample_idx, ...].

        Returns
        -------
        states: np.array of shape (T, d, k)
            The most likely hidden state trajectories in binary vector form,
            given the observable time series.

        Notes
        -----
        Complexity :math:`\mathcal{O}(Tdk^{d+1})`.
        """
        # eps = np.finfo(np.float32).eps
        eps = 0

        realizations = self.realizations()
        py = self.py(sample_idx)

        # Initialize delta (just like alpha)
        delta = np.zeros(shape=(self.T, self.k**self.d))
        psi = np.zeros(shape=(self.T, self.k**self.d), dtype=int)
        for i in range(realizations.shape[1]):
            pi = 1
            for d in range(self.d):
                pi *= self.pi[d, realizations[d, i]]

            delta[0, i] = pi * py[0, i] + eps
            psi[0, i] = 0

        # Recursion
        for t in range(1, self.T):
            for j in range(realizations.shape[1]):
                prob_j = 1
                for d in range(self.d):
                    prob_j *= np.exp(self.A)[d, realizations[d, j], realizations[d, :]]
                delta[t, j] = np.max(delta[t-1] * prob_j) * py[t, j] + eps
                psi[t, j] = np.argmax(delta[t-1] * prob_j)

            delta[t, :] /= delta[t, :].sum()

        p_star = np.max(delta[self.T-1])

        # Backtrack through the most likely states
        q_star = np.zeros(shape=self.T, dtype=int)
        q_star[self.T-1] = int(np.argmax(delta[self.T-1]))

        for t in reversed(range(self.T-1)):
            q_star[t] = int(psi[t+1, q_star[t+1]])

        # transform `q` to states
        states = np.zeros(shape=(self.T, self.d, self.k))

        for t in range(self.T):
            for d in range(self.d):
                states[t, d, realizations[d, q_star[t]]] = 1

        return states

    def generate(self, n_samples, length, return_states=False):
        """ Use to create a dataset based on currently set model parameters.

        Parameters
        ----------
        n_samples: int
            Number of random time series examples to generate.
        length: int
            Length, as number of samples, of each generated time series.
        return_states: bool, False
            Whether or not to return a matrix containing all of the hidden
            state values for all of the samples.

        Returns
        -------

        X: np.array of shape (N, T, o)
            The generated time series example(s).

        states: np.array of shape (N, T, d, k)
            All of the hidden state values in binary vector form.
        """
        X = np.zeros(shape=(n_samples, length, self.o))

        if return_states:
            states = np.zeros(shape=(n_samples, length, self.d, self.k))
        else:
            states = None

        # Hidden states
        s = np.ndarray(shape=(length, self.d, self.k))

        for n in range(n_samples):
            y = np.zeros(shape=(length, self.o))
            for t in range(length):
                for d in range(self.d):
                    if t == 0:
                        p = self.pi[d]
                    else:
                        p = np.exp(self.A[d]).dot(s[t-1, d, :])  # Transition probability

                    # Randomly sample from categorical, transition distribution
                    idx = np.random.choice(range(self.k), p=p)
                    s[t, d, :] = 0
                    s[t, d, idx] = 1

                y_mu = np.einsum('dok,dk', self.W, s[t, :, :])

                # Sample from emission distribution
                y[t] = np.random.multivariate_normal(y_mu, self.C)

            X[n] = y.copy()

            if return_states:
                states[n] = s.copy()

        return X, states

    @classmethod
    def generate_random_model_params(cls, T, d, k, o, seed=None):
        """ Method to generate random model parameters: W, A, C, pi.

        Parameters
        ----------
        T : int, (Z+)
            The length of each sequence.
        d : int, (Z+)
            The number of hidden vectors, at each time step.
        k : int, (Z+)
            The length of the hidden vectors -- number of states.
        o : int, (Z+)
            The length of the output vector.
        seed: int, None
            To fix the random seed.

        Returns
        -------
        W, A, C, pi: tuple
            Model parameters.

        """
        if seed is not None:
            np.random.seed(seed)
        C = cls.generate_random_C(o)
        A = cls.generate_random_A(d, k, k)
        W = np.random.uniform(0, 1, size=k*o*d).reshape(d, o, k)
        pi = cls.generate_random_pi(d, k)
        return W, A, C, pi

    def plot_fit(self, sample_idx=None, include_noise=True, ax=None):
        r""" Plot the time series along with the most likely model output
        built using the hidden state trajectories from the Viterbi algorithm.

        Parameters
        ----------
        sample_idx: int, None
            The specific example to plot via selecting X[sample_idx, ...].

        include_noise: bool, True
            Whether or not to add the model estimated noise to the discrete
            levels when plotting.

        ax: matplotlib.axes.Axes, None
            A figure on which to plot a single example.  Must have sample_idx
            specified.

        Returns
        -------
        axs: list of Axes objects
            A list of the figures for each time series example.

        Notes
        -----
        Complexity :math:`\mathcal{O}(NTdk^{d+1})`.
        """

        if sample_idx is None:
            sample_indices = range(self.X.shape[0])
        else:
            sample_indices = [sample_idx]

        axs = []
        for sample_idx in sample_indices:
            vstates = self.viterbi(sample_idx)
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            else:
                assert len(sample_indices) == 1

            label = 'Data Sample {}'.format(sample_idx)
            ax.plot(self.X[sample_idx], label=label)

            y = np.zeros(shape=(self.T, self.o))
            for t in range (self.T):
                y_mu = np.einsum('dok,dk', self.W, vstates[t])

                if include_noise:
                    y[t] = np.random.multivariate_normal(y_mu, self.C)
                else:
                    C = np.zeros(shape=self.C.shape)
                    y[t] = np.random.multivariate_normal(y_mu, C)

            ax.plot(range(self.T), y.ravel(), label='Fit')

            ax.set_ylabel('Signal (Arbitrary Units)', fontsize=15)
            ax.set_xlabel('Time (Arbitrary Units)', fontsize=15)

            plt.legend()

            axs.append(ax)

        return axs


class FHMMCV(FHMM):
    """Factorial Hidden Markov Model (FHMM) implementation with time-wise
    train-test splitting. Uses log likelihood scoring.

    Parameters
    ----------
    T : int, (Z+)
        The length of each sequence.
    d : int, (Z+)
        The number of hidden vectors, at each time step.
    k : int, (Z+)
        The length of the hidden vectors -- number of states.
    o : int, (Z+)
        The length of the output vector.
    n_restarts : int, (Z+)
        Number of full model restarts, in search of the global optimum.
    em_max_iter : int, (Z+)
        Maximum number of cycles through E-M steps.
    em_log_likelihood_tol : float, default=1E-8
        The tolerance level to discern one likelihood value to the next.
    em_log_likelihood_count : int, (Z+)
        Number of log likelihood values without change (according to
        `em_log_likelihood_tol`) indicating convergence.
    e_step_retries : int, (Z+)
        Number of random restarts of (applicable) E method.
    method : str, ('gibbs', 'mean_field', 'sva', 'exact')
        Selecting the method for expectation maximization. Options
        are 'gibbs' for Gibbs sampling, 'mean_field' for
        using mean field estimation (or completely factorized approximation), and
        'sva' to use the Structured Variational Approximation (SVA), and
        'exact' for the exact solve (very slow -- do not use for large parameter values!).
    gibbs_max_iter : int, (Z+)
        Number of states sampled within Gibbs E-step.
    mean_field_max_iter : int, (Z+)
        Maximum number of mean field updates.  Once reached, will
        exit without necessarily meeting KLD tolerance.
    mean_field_kld_tol : float, (R+)
        Tolerance for change in KLD between mean field iterations.
    sva_max_iter : int, (Z+)
        Maximum number of Structured Variational Approximation (SVA) updates.  Once reached, will
        exit without necessarily meeting KLD tolerance.
    sva_kld_tol : float, (R+)
        Tolerance for change in KLD between SVA iterations.
    stochastic_training : bool
        Whether or not to use stochastic training -- random and decaying jostling of fit parameters while learning.
    stochastic_lr : float
        Roughly the size of the random excursions in fit parameters.
    zero_probability : float, (R+)
        Numerical cutoff indicating zero probability (not strictly zero).
    W_init : numpy.array, None
        Initialize the starting W weight matrix (shape (T, d, k)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial W.
    A_init : numpy.array, None
        Initialize the starting A transition matrix (shape=(d, k, k)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial A.
    C_init : numpy.array, None
        Initialize the starting C covariance matrix (shape=(o, o)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial C.
    pi_init : numpy.array, None
        Initialize the starting pi initial state distribution matrix (shape=(d, k)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial pi.
    W_fixed : numpy.array, None
        Set equal to the true W weight matrix (shape (T, d, k)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update W.
    A_fixed : numpy.array, None
        Set equal to the true A transition matrix (shape=(d, k, k)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update A.
    C_fixed : numpy.array, None
        Set equal to the true C covariance matrix (shape=(o, o)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update C.
    pi_fixed : numpy.array, None
        Set equal to the true pi initial state distribution matrix (shape=(d, k)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update pi.
    verbose : bool, True
        Print progress and possibly other indicators of algorithm state.
    test_size : float, (0, 1)
        Test set fraction of subsequence_size
    subsequence_size : float, (0, 1)
        Fraction of total data, giving the length of portions for train/test
    n_splits : int, (Z+)
        Number of `subsequence_size` portions to use for fitting
        iterations.
    n_jobs : int, >0 or -1
        Number of parallel processes to use for computation, must be
        greater than zero, or equal to -1, indicating to use all
        resources.


    Examples
    --------
    >>> parameters = {'T': X.shape[1],
    ...               'o': X.shape[2],
    ...               'd': 4,   # Number of hidden vectors at each time 't'
    ...               'k': 8}   # Dimension of the hidden state vectors
    >>> fhmmcv = FHMMCV(**parameters)
    >>> fhmmcv.fit(X)
    >>> params = fhmmcv.best_params

    Attributes
    ----------
    scores: list
        List of scores from each CV round.
    params: list
        Corresponding list of parameters found in each CV round.
    best_score: float
        The best scoring CV fit score.
    best_params: dict
        The best scoring CV fit parameters, as a dictionary appropriate for
        self.set_params(**best_params).

    """

    def __init__(self,
                 T=10,
                 d=4,
                 k=8,
                 o=4,
                 n_restarts=5,
                 em_max_iter=10,
                 em_log_likelihood_tol=1E-8,
                 em_log_likelihood_count=4,
                 e_step_retries=0,
                 method='sva',
                 gibbs_max_iter=200,
                 mean_field_max_iter=10,
                 mean_field_kld_tol=1E-8,
                 sva_max_iter=10,
                 sva_kld_tol=1E-8,
                 stochastic_training=False,
                 stochastic_lr=0.1,
                 zero_probability=1E-8,
                 W_init=None,
                 A_init=None,
                 C_init=None,
                 pi_init=None,
                 W_fixed=None,
                 A_fixed=None,
                 C_fixed=None,
                 pi_fixed=None,
                 save_directory=None,
                 save_x=False,
                 verbose=True,
                 test_size=0.2,
                 subsequence_size=0.3,
                 n_splits=3,
                 n_jobs=-1):

        super().__init__(T=T,
                         d=d,
                         k=k,
                         o=o,
                         n_restarts=n_restarts,
                         em_max_iter=em_max_iter,
                         em_log_likelihood_tol=em_log_likelihood_tol,
                         em_log_likelihood_count=em_log_likelihood_count,
                         e_step_retries=e_step_retries,
                         method=method,
                         gibbs_max_iter=gibbs_max_iter,
                         mean_field_max_iter=mean_field_max_iter,
                         mean_field_kld_tol=mean_field_kld_tol,
                         sva_max_iter=sva_max_iter,
                         sva_kld_tol=sva_kld_tol,
                         stochastic_training=stochastic_training,
                         stochastic_lr=stochastic_lr,
                         zero_probability=zero_probability,
                         W_init=W_init,
                         A_init=A_init,
                         C_init=C_init,
                         pi_init=pi_init,
                         W_fixed=W_fixed,
                         A_fixed=A_fixed,
                         C_fixed=C_fixed,
                         pi_fixed=pi_fixed,
                         save_directory=save_directory,
                         save_x=save_x,
                         verbose=verbose)

        self.test_size = test_size
        self.subsequence_size = subsequence_size
        self.n_splits = n_splits
        self.n_jobs = n_jobs

        self.scores = []
        self.params = []

        self.best_score = None
        self.best_params = None

    def time_splits(self, X):
        """ Generates time-wise splits of the data matrix X, based on the
        settings of `subsequence_length`, `test_size`, and `n_splits`.
        Splits X along axis=1 into (possibly overlapping) `n_splits` segments
        of `subsequence_length` length; then further splits the subsequence
        into train/test specified by the `test_size` fraction, yielding the
        train test data at each iteration.

        Parameters
        ----------
        X: np.array of shape (N, T, o)

        Yields
        ------
        X_train, X_test: tuple
            Current iteration's (time-wise split) train/test data.
        """
        self.subsequence_length_ = int(np.floor(X.shape[1] * self.subsequence_size))
        t_offset = (X.shape[1] - self.subsequence_length_) // self.n_splits-1

        self.train_length_ = int(np.floor(self.subsequence_length_ * (1-self.test_size)))
        self.test_length_ = self.subsequence_length_ - self.train_length_
        self.start_indices_ = []
        for split in range(self.n_splits):
            start_idx = split * t_offset
            self.start_indices_.append(start_idx)
            end_idx = start_idx + self.train_length_
            X_train = X[:, start_idx:end_idx+1, :]  # NOTE: 1 output of overlap for initial condition
            X_test = X[:, end_idx:end_idx+self.test_length_, :]

            yield X_train, X_test

    def _build_run_directory(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return os.path.join(self.save_directory, timestamp+"_crossval")
    
    def fit(self, X):
        """ Multiple rounds of cross validation fits, storing results in
        best_params and best_scores class attributes.

        Parameters
        ----------
        X: numpy.array of shape (N, T, o)
            The input data.

        Returns
        -------
        self: nomopy.fhmm.FHMM
            An instance of FHMM with parameters set to the best fitting
            values, and best_params and best_scores class attributes
            populated.
        """
        self.X = X
        if self._is_fixed_warn(): return self
        self._create_run_directory_and_save_x_if_needed()

        def fit_train_test(X_train_t, X_test_t, i):
            cv_params = self.get_params()

            train_params = {}
            fhmm_params = FHMM().get_params()
            for k, v in cv_params.items():
                if k in fhmm_params.keys():
                    train_params[k] = v

            fhmm_train = FHMM(**train_params)
            fhmm_train._run_directory = os.path.join(self._run_directory, "time_split_"+str(i))
            fhmm_train.fit(X_train_t)

            train_params = fhmm_train.get_params()

            # Estimate final state probability (as initial probability for test set)
            pi_fixed = fhmm_train.final_state_distribution()

            # Score on testset
            fhmm_test = FHMM(**train_params)
            fhmm_test.pi = fhmm_test.pi_fixed = pi_fixed   # Fix the initial distribution

            # Find the state expectation values
            fhmm_test.X = X_test_t
            fhmm_test._update_expectations()

            score = fhmm_test.log_likelihood()

            return (score, fhmm_test.get_params())

        # futures = []
        # for X_train_t, X_test_t in self.time_splits(X):
        #     futures.append(delayed(fit_train_test)(X_train_t, X_test_t))

        # outputs = Parallel(n_jobs=self.n_jobs)(futures)

        outputs = []
        for i, (X_train_t, X_test_t) in enumerate(self.time_splits(X)):
            outputs.append(fit_train_test(X_train_t, X_test_t, i))

        for output in outputs:
            self.scores.append(output[0])
            self.params.append(output[1])

        idx_best = np.argmax(self.scores)
        self.best_score = self.scores[idx_best]
        self.best_params = self.params[idx_best]

        ####################
        # Set to best model
        best_params = self.get_params()
        fhmm_params = FHMM().get_params()
        for k, v in self.get_params().items():
            if k in fhmm_params:
                best_params[k] = self.best_params[k]
        self.set_params(**best_params)

        # Set dataset
        for idx, (X_train_t, X_test_t) in enumerate(self.time_splits(X)):
            if idx == idx_best:
                self.X = X_test_t

        # Set fit params
        self.W = self.W_fixed
        self.A = self.A_fixed
        self.C = self.C_fixed
        self.pi = self.pi_fixed

        # Update expectationss
        self._update_expectations()

        return self

    @classmethod
    def load_models_for_all_splits(cls, save_directory, load_x_from_snapshot=False, x_filepath=None):
        """Load models for all CV time splits

        Parameters
        ----------
        save_directory : _type_
            _description_
        load_x_from_snapshot : bool, optional
            _description_, by default False
        x_filepath : _type_, optional
            _description_, by default None
        """
        split_directories = [os.path.join(save_directory, f) for f in os.listdir(save_directory) if not os.path.isfile(os.path.join(save_directory, f))]
        split_numbers = [int(d.split('_')[-1]) for d in split_directories]
        models = FHMM.load_best_models_from_directories(split_directories, load_x_from_snapshot=load_x_from_snapshot, x_filepath=x_filepath)
        return {split_number: model for (split_number,model) in zip(split_numbers, models)}




class FHMMBS(FHMM):
    """Factorial Hidden Markov Model (FHMM) implementation with bootstraping.
     Uses log likelihood scoring.

    Parameters
    ----------
    T : int, (Z+)
        The length of each sequence.
    d : int, (Z+)
        The number of hidden vectors, at each time step.
    k : int, (Z+)
        The length of the hidden vectors -- number of states.
    o : int, (Z+)
        The length of the output vector.
    n_restarts : int, (Z+)
        Number of full model restarts, in search of the global optimum.
    em_max_iter : int, (Z+)
        Maximum number of cycles through E-M steps.
    em_log_likelihood_tol : float, default=1E-8
        The tolerance level to discern one likelihood value to the next.
    em_log_likelihood_count : int, (Z+)
        Number of log likelihood values without change (according to
        `em_log_likelihood_tol`) indicating convergence.
    e_step_retries : int, (Z+)
        Number of random restarts of (applicable) E method.
    method : str, ('gibbs', 'mean_field', 'sva', 'exact')
        Selecting the method for expectation maximization. Options
        are 'gibbs' for Gibbs sampling, 'mean_field' for
        using mean field estimation (or completely factorized approximation), and
        'sva' to use the Structured Variational Approximation (SVA), and
        'exact' for the exact solve (very slow -- do not use for large parameter values!).
    gibbs_max_iter : int, (Z+)
        Number of states sampled within Gibbs E-step.
    mean_field_max_iter : int, (Z+)
        Maximum number of mean field updates.  Once reached, will
        exit without necessarily meeting KLD tolerance.
    mean_field_kld_tol : float, (R+)
        Tolerance for change in KLD between mean field iterations.
    sva_max_iter : int, (Z+)
        Maximum number of Structured Variational Approximation (SVA) updates.  Once reached, will
        exit without necessarily meeting KLD tolerance.
    sva_kld_tol : float, (R+)
        Tolerance for change in KLD between SVA iterations.
    stochastic_training : bool
        Whether or not to use stochastic training -- random and decaying jostling of fit parameters while learning.
    stochastic_lr : float
        Roughly the size of the random excursions in fit parameters.
    zero_probability : float, (R+)
        Numerical cutoff indicating zero probability (not strictly zero).
    W_init : numpy.array, None
        Initialize the starting W weight matrix (shape (T, d, k)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial W.
    A_init : numpy.array, None
        Initialize the starting A transition matrix (shape=(d, k, k)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial A.
    C_init : numpy.array, None
        Initialize the starting C covariance matrix (shape=(o, o)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial C.
    pi_init : numpy.array, None
        Initialize the starting pi initial state distribution matrix (shape=(d, k)), to provide estimation
        a good starting point.  Can be used for debugging, or warm starting.  If `None`, algorithm will choose
        an initial pi.
    W_fixed : numpy.array, None
        Set equal to the true W weight matrix (shape (T, d, k)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update W.
    A_fixed : numpy.array, None
        Set equal to the true A transition matrix (shape=(d, k, k)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update A.
    C_fixed : numpy.array, None
        Set equal to the true C covariance matrix (shape=(o, o)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update C.
    pi_fixed : numpy.array, None
        Set equal to the true pi initial state distribution matrix (shape=(d, k)), to bypass estimation.  Can be
        used for debugging.  If `None`, algorithm will update pi.
    verbose : bool, True
        Print progress and possibly other indicators of algorithm state.
    sample_size : int, (Z+)
        Size of bootstrap sample
    n_bootstrap_samples : int, (Z+)
        Number of bootstrap samples to fit
    n_jobs : int, >0 or -1
        Number of parallel processes to use for computation, must be
        greater than zero, or equal to -1, indicating to use all
        resources.


    Examples
    --------
    >>> parameters = {'T': X.shape[1],
    ...               'o': X.shape[2],
    ...               'd': 4,   # Number of hidden vectors at each time 't'
    ...               'k': 8}   # Dimension of the hidden state vectors
    >>> fhmmbs = FHMMBS(**parameters)
    >>> fhmmbs.fit(X)

    Attributes
    ----------


    """

    def __init__(self,
                 T=10,
                 d=4,
                 k=8,
                 o=4,
                 n_restarts=5,
                 em_max_iter=10,
                 em_log_likelihood_tol=1E-8,
                 em_log_likelihood_count=4,
                 e_step_retries=0,
                 method='sva',
                 gibbs_max_iter=200,
                 mean_field_max_iter=10,
                 mean_field_kld_tol=1E-8,
                 sva_max_iter=10,
                 sva_kld_tol=1E-8,
                 stochastic_training=False,
                 stochastic_lr=0.1,
                 zero_probability=1E-8,
                 W_init=None,
                 A_init=None,
                 C_init=None,
                 pi_init=None,
                 W_fixed=None,
                 A_fixed=None,
                 C_fixed=None,
                 pi_fixed=None,
                 save_directory=None,
                 save_x=False,
                 verbose=True,
                 sample_size=100,
                 n_bootstrap_samples=10,
                 n_jobs=-1
                 ):

        super().__init__(T=T,
                         d=d,
                         k=k,
                         o=o,
                         n_restarts=n_restarts,
                         em_max_iter=em_max_iter,
                         em_log_likelihood_tol=em_log_likelihood_tol,
                         em_log_likelihood_count=em_log_likelihood_count,
                         e_step_retries=e_step_retries,
                         method=method,
                         gibbs_max_iter=gibbs_max_iter,
                         mean_field_max_iter=mean_field_max_iter,
                         mean_field_kld_tol=mean_field_kld_tol,
                         sva_max_iter=sva_max_iter,
                         sva_kld_tol=sva_kld_tol,
                         stochastic_training=stochastic_training,
                         stochastic_lr=stochastic_lr,
                         zero_probability=zero_probability,
                         W_init=W_init,
                         A_init=A_init,
                         C_init=C_init,
                         pi_init=pi_init,
                         W_fixed=W_fixed,
                         A_fixed=A_fixed,
                         C_fixed=C_fixed,
                         pi_fixed=pi_fixed,
                         save_directory=save_directory,
                         save_x=save_x,
                         verbose=verbose)
        self.sample_size = sample_size
        self.n_bootstrap_samples = n_bootstrap_samples
        self.n_jobs = n_jobs
        
        self.results = None

    def sample_generator(self):
        for _ in range(self.n_bootstrap_samples):
            start_idx = np.random.choice(range(self.X.shape[1] - self.sample_size))
            end_idx = start_idx + self.sample_size

            yield self.X[:, start_idx:end_idx, :]


    def _create_param_dict_for_bootstrap_instances(self):

        param_dict = FHMM().get_params()
        bs_param_dict = self.get_params()

        for param in param_dict.keys():
            param_dict[param] = bs_param_dict[param]

        if self.save_directory:
            param_dict['save_directory'] = self._run_directory


        return param_dict

    def _build_run_directory(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return os.path.join(self.save_directory, timestamp+"_bootstraping")

    
    def fit(self, X):

        self.X = X
        if self._is_fixed_warn(): return self
        self._create_run_directory_and_save_x_if_needed()

        assert 0 < self.sample_size <= self.X.shape[1], "Sample size must be between 0 and {}".format(self.X.shape[1])
                
        self.results = []
        for i, X_sample in enumerate(self.sample_generator()):
            result = self._fit_bootstrap(i, X_sample)
            self.results.append(result)

        self._extract_fit_params_from_bootstraps()
        self._sort_weights_and_transitions()

    def _fit_bootstrap(self, i, X_sample):
        param_dict = self._create_param_dict_for_bootstrap_instances()
        fhmm_bootstrap = FHMM(**param_dict)
        fhmm_bootstrap._run_directory = os.path.join(self._run_directory,
                                                         "bootstrap_sample_"+str(i))
        if self.verbose:
            print("---- Bootstrap #{} ----".format(i))
        fhmm_bootstrap.fit(X_sample)
        result = (fhmm_bootstrap.score(), fhmm_bootstrap.get_params())
        return result

    def _extract_fit_params_from_bootstraps(self):
        Ws = []
        As = []
        Cs = []
        pis = []
        
        for _, result in self.results:
            Ws.append(result['W_fixed'])
            As.append(result['A_fixed'])
            Cs.append(result['C_fixed'])
            pis.append(result['pi_fixed'])

        self.W_bootstraps = np.array(Ws)
        self.A_bootstraps = np.array(As)
        self.C_bootstraps = np.array(Cs)
        self.pi_bootstraps = np.array(pis)
    
    def _sort_weights_and_transitions(self):
        # Placeholder function to sort weights so each bootstrap matches
        pass

    def get_average_parameters(self):
        assert self.results is not None

        W_average = np.mean(self.W_bootstraps, axis=0)
        A_average = np.mean(self.A_bootstraps, axis=0)
        C_average = np.mean(self.C_bootstraps, axis=0)
        pi_average = np.mean(self.pi_bootstraps, axis=0)

        return W_average, A_average, C_average, pi_average

    @classmethod
    def load_models_for_all_bootstraps(cls, save_directory, load_x_from_snapshot=False, x_filepath=None):
        """Load models for all bootstraps

        Parameters
        ----------
        save_directory : _type_
            _description_
        load_x_from_snapshot : bool, optional
            _description_, by default False
        x_filepath : _type_, optional
            _description_, by default None
        """
        bootstrap_directories = [os.path.join(save_directory, f) for f in os.listdir(save_directory) if not os.path.isfile(os.path.join(save_directory, f))]
        bootstrap_numbers = [int(d.split('_')[-1]) for d in bootstrap_directories]
        models = FHMM.load_best_models_from_directories(bootstrap_directories, load_x_from_snapshot=load_x_from_snapshot, x_filepath=x_filepath)
        return {bootstrap_number: model for (bootstrap_number,model) in zip(bootstrap_numbers, models)}
