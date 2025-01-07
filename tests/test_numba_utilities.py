import os
import pickle

import numpy as np
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
import pickle





TEST_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def get_random_model_params(d, k, o):
    W = np.random.uniform(0, 1, size=k*o*d).reshape(d, o, k)
    A = np.random.uniform(1E-9, 1, size=k*k*d).reshape(d, k, k)
    A /= A.sum(axis=1, keepdims=True)
    A = np.log(A)
    C = np.random.uniform(0, 1, size=o**2).reshape(o, o)
    C = (C.T).dot(C)
    pi = np.random.uniform(1E-9, 1, size=d*k).reshape(d, k)
    pi /= pi.sum(axis=1, keepdims=True)

    return W, A, C, pi

def adjust_random_parameters(TdokN):
    T, d, o, k, N = TdokN
    T *= 5
    N *= 5
    return T, d, o, k, N

  
@pytest.mark.slow
def test_exact_fit_runs_no_jit(monkeypatch):
    monkeypatch.setenv("PYTEST_DISABLE_JIT", "1")
    from nomopy.fhmm import FHMM

    def get_random_data(T, d, k, o, N):
        W, A, C, pi = get_random_model_params(d, k, o)
        fhmm = FHMM(T=T,
                    d=d,
                    k=k,
                    o=o,
                    W_fixed=W,
                    A_fixed=A,
                    C_fixed=C,
                    pi_fixed=pi)
        return fhmm.generate(N, T)
 
    TdokN = (3, 3, 3, 3, 3)
    T, d, o, k, N = adjust_random_parameters(TdokN)
    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMM(T=T,
                d=d,
                k=k,
                o=o,
                n_restarts=0,
                em_max_iter=2,
                method='exact',
                verbose=False)

    fhmm.fit(X)