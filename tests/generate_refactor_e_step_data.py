import sys
sys.path.append('./')
import pickle

import numpy as np

from nomopy.fhmm import FHMM


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


def adjust_random_parameters(TdokN):
    T, d, o, k, N = TdokN
    T *= 5
    N *= 5
    return T, d, o, k, N


if __name__ == '__main__':
    method = str(sys.argv[1])

    TdokN = (3, 3, 3, 3, 3)

    np.random.seed(0)
    T, d, o, k, N = adjust_random_parameters(TdokN)
    X, _ = get_random_data(T, d, k, o, N)

    np.random.seed(1)
    W, A, C, pi = get_random_model_params(d, k, o)
    # Test fit iterations on random data
    fhmm = FHMM(T=T,
                d=d,
                k=k,
                o=o,
                W_fixed=W,
                A_fixed=A,
                C_fixed=C,
                pi_fixed=pi,
                n_restarts=0,
                em_max_iter=2,
                method=method,
                verbose=False)

    fhmm.X = X

    np.random.seed(2)
    s_exp, ss_exp, sstm1_exp = fhmm.E(0)

    fit_data = {'W': W,
                'A': A,
                'C': C,
                'pi': pi,
                'X': X,
                's_exp': s_exp,
                'ss_exp': ss_exp,
                'sstm1_exp': sstm1_exp}

    with open(f'./tests/refactor_data/random_{method}_e_step_data.pkl', 'wb') as f:
        pickle.dump(fit_data, f)
