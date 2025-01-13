import os
import pickle
import time

import numpy as np
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from nomopy.fhmm import FHMM, FHMMCV, FHMMBS


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


def test_initialize_parameters_none():
    T, d, o, k = 1, 1, 1, 2

    fhmm = FHMM(T=T, d=d, o=o, k=k)

    assert np.prod(fhmm.W.shape) == d*o*k
    assert np.prod(fhmm.A.shape) == d*k*k
    assert np.prod(fhmm.C.shape) == o*o
    assert np.prod(fhmm.pi.shape) == d*k


def test_initialize_parameters_initial():
    T, d, o, k = 1, 1, 1, 2

    W_init = np.random.uniform(0, 1, size=k*o*d).reshape(d, o, k)
    A_init = np.random.uniform(1E-9, 1, size=k*k*d).reshape(d, k, k)
    A_init /= A_init.sum(axis=1, keepdims=True)
    A_init = np.log(A_init)
    C_init = np.random.uniform(0, 1, size=o**2).reshape(o, o)
    C_init = (C_init.T).dot(C_init)
    pi_init = np.random.uniform(1E-9, 1, size=d*k).reshape(d, k)
    pi_init /= pi_init.sum(axis=1, keepdims=True)

    fhmm = FHMM(T=T, d=d, o=o, k=k,
                W_init=W_init,
                A_init=A_init,
                C_init=C_init,
                pi_init=pi_init)

    assert np.all(fhmm.W == fhmm.W_init)
    assert np.all(fhmm.A == fhmm.A_init)
    assert np.all(fhmm.C == fhmm.C_init)
    assert np.all(fhmm.C_inv == np.linalg.inv(fhmm.C_init))
    assert np.all(fhmm.pi == fhmm.pi_init)

    fhmm.W += 1
    fhmm.A += 1
    mat = np.random.uniform(0, 1, size=o**2).reshape(o, o)
    fhmm.C = (mat.T).dot(mat)
    fhmm.pi += 1

    assert np.all(fhmm.W != fhmm.W_init)
    assert np.all(fhmm.A != fhmm.A_init)
    assert np.all(fhmm.C != fhmm.C_init)
    assert np.all(fhmm.C_inv != np.linalg.inv(fhmm.C_init))
    assert np.all(fhmm.pi != fhmm.pi_init)

    assert np.all(fhmm.C_inv == np.linalg.inv(fhmm.C))


def test_initialize_parameters_fixed_sets_params():
    T, d, o, k = 1, 1, 1, 2

    W_fixed = np.random.uniform(0, 1, size=k*o*d).reshape(d, o, k)
    A_fixed = np.random.uniform(1E-9, 1, size=k*k*d).reshape(d, k, k)
    A_fixed /= A_fixed.sum(axis=1, keepdims=True)
    A_fixed = np.log(A_fixed)
    C_fixed = np.random.uniform(0, 1, size=o**2).reshape(o, o)
    C_fixed = (C_fixed.T).dot(C_fixed)
    pi_fixed = np.random.uniform(1E-9, 1, size=d*k).reshape(d, k)
    pi_fixed /= pi_fixed.sum(axis=1, keepdims=True)

    fhmm = FHMM(T=T, d=d, o=o, k=k,
                W_fixed=W_fixed,
                A_fixed=A_fixed,
                C_fixed=C_fixed,
                pi_fixed=pi_fixed)

    assert np.all(fhmm.W == fhmm.W_fixed)
    assert np.all(fhmm.A == fhmm.A_fixed)
    assert np.all(fhmm.C == fhmm.C_fixed)
    assert np.all(fhmm.C_inv == np.linalg.inv(fhmm.C_fixed))
    assert np.all(fhmm.pi == fhmm.pi_fixed)


def test_initialize_parameters_fixed_updates_params():
    T, d, o, k = 1, 1, 1, 2

    W_fixed = np.random.uniform(0, 1, size=k*o*d).reshape(d, o, k)
    A_fixed = np.random.uniform(1E-9, 1, size=k*k*d).reshape(d, k, k)
    A_fixed /= A_fixed.sum(axis=1, keepdims=True)
    A_fixed = np.log(A_fixed)
    C_fixed = np.random.uniform(0, 1, size=o**2).reshape(o, o)
    C_fixed = (C_fixed.T).dot(C_fixed)
    pi_fixed = np.random.uniform(1E-9, 1, size=d*k).reshape(d, k)
    pi_fixed /= pi_fixed.sum(axis=1, keepdims=True)

    fhmm = FHMM(T=T, d=d, o=o, k=k,
                W_fixed=W_fixed,
                A_fixed=A_fixed,
                C_fixed=C_fixed,
                pi_fixed=pi_fixed)

    fhmm.W_fixed += 1
    fhmm.A_fixed += 1
    mat = np.random.uniform(0, 1, size=o**2).reshape(o, o)
    fhmm.C_fixed = (mat.T).dot(mat)
    fhmm.pi_fixed += 1

    assert np.all(fhmm.W == fhmm.W_fixed)
    assert np.all(fhmm.A == fhmm.A_fixed)
    assert np.all(fhmm.C == fhmm.C_fixed)
    assert np.all(fhmm.C_inv == np.linalg.inv(fhmm.C_fixed))
    assert np.all(fhmm.pi == fhmm.pi_fixed)


def test_initialize_parameters_fixed_keeps_params_unchanged():
    T, d, o, k = 1, 1, 1, 2

    W_fixed = np.random.uniform(0, 1, size=k*o*d).reshape(d, o, k)
    A_fixed = np.random.uniform(1E-9, 1, size=k*k*d).reshape(d, k, k)
    A_fixed /= A_fixed.sum(axis=1, keepdims=True)
    A_fixed = np.log(A_fixed)
    C_fixed = np.random.uniform(0, 1, size=o**2).reshape(o, o)
    C_fixed = (C_fixed.T).dot(C_fixed)
    pi_fixed = np.random.uniform(1E-9, 1, size=d*k).reshape(d, k)
    pi_fixed /= pi_fixed.sum(axis=1, keepdims=True)

    fhmm = FHMM(T=T, d=d, o=o, k=k,
                W_fixed=W_fixed,
                A_fixed=A_fixed,
                C_fixed=C_fixed,
                pi_fixed=pi_fixed)

    fhmm.W += 1
    fhmm.A += 1
    mat = np.random.uniform(0, 1, size=o**2).reshape(o, o)
    fhmm.C = (mat.T).dot(mat)
    fhmm.pi += 1

    assert np.all(fhmm.W == fhmm.W_fixed)
    assert np.all(fhmm.A == fhmm.A_fixed)
    assert np.all(fhmm.C == fhmm.C_fixed)
    assert np.all(fhmm.pi == fhmm.pi_fixed)
    assert np.all(fhmm.C_inv == np.linalg.inv(fhmm.C_fixed))


def test_X_mismatch_raises():
    T, d, o, k = 1, 1, 1, 2

    W_fixed = np.random.uniform(0, 1, size=k*o*d).reshape(d, o, k)
    A_fixed = np.random.uniform(1E-9, 1, size=k*k*d).reshape(d, k, k)
    A_fixed /= A_fixed.sum(axis=1, keepdims=True)
    A_fixed = np.log(A_fixed)
    C_fixed = np.random.uniform(0, 1, size=o**2).reshape(o, o)
    C_fixed = (C_fixed.T).dot(C_fixed)
    pi_fixed = np.random.uniform(1E-9, 1, size=d*k).reshape(d, k)
    pi_fixed /= pi_fixed.sum(axis=1, keepdims=True)

    fhmm = FHMM(T=T, d=d, o=o, k=k,
                W_fixed=W_fixed,
                A_fixed=A_fixed,
                C_fixed=C_fixed,
                pi_fixed=pi_fixed)

    X = np.random.rand(10, T, 2)

    with pytest.raises(ValueError):
        fhmm.X = X


def test_set_params_C_inv_uptodate():
    T, d, o, k = 1, 1, 1, 2

    W_fixed = np.random.uniform(0, 1, size=k*o*d).reshape(d, o, k)
    A_fixed = np.random.uniform(1E-9, 1, size=k*k*d).reshape(d, k, k)
    A_fixed /= A_fixed.sum(axis=1, keepdims=True)
    A_fixed = np.log(A_fixed)
    C_fixed = np.random.uniform(0, 1, size=o**2).reshape(o, o)
    C_fixed = (C_fixed.T).dot(C_fixed)
    pi_fixed = np.random.uniform(1E-9, 1, size=d*k).reshape(d, k)
    pi_fixed /= pi_fixed.sum(axis=1, keepdims=True)

    fhmm = FHMM(T=T, d=d, o=o, k=k)

    params = fhmm.get_params()

    params.update({'C_fixed': C_fixed})

    fhmm.set_params(**params)

    assert np.all(np.linalg.inv(fhmm.C) == fhmm.C_inv)


@pytest.mark.slow
@settings(max_examples=5, deadline=None)
@given(arrays(np.int32, 5, elements=st.integers(1, 3)))
def test_canonically_transform_gives_same_ll(TdokN):
    T, d, o, k, N = adjust_random_parameters(TdokN)
    W, A, C, pi = FHMM.generate_random_model_params(T, d, k, o)

    fhmm = FHMM(T=T,
                d=d,
                k=k,
                o=o,
                W_fixed=W,
                A_fixed=A,
                C_fixed=C,
                pi_fixed=pi,
                method='mean_field')
    X, _ = fhmm.generate(N, T)
    fhmm.X = X

    ll_before = fhmm.log_likelihood()
    W_new = fhmm.canonically_transform_W(W)
    fhmm.W_fixed = W_new
    ll_after = fhmm.log_likelihood()

    assert ll_after == pytest.approx(ll_before, 1E-6)
    assert np.all(fhmm.W_fixed == W_new)


@pytest.mark.slow
@settings(max_examples=1, deadline=None)
@given(arrays(np.int32, 5, elements=st.integers(1, 3)))
def test_exact_fit_runs(TdokN):
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


@pytest.mark.slow
@settings(max_examples=1, deadline=None)
@given(arrays(np.int32, 5, elements=st.integers(1, 3)))
def test_mean_field_fit_runs(TdokN):
    T, d, o, k, N = adjust_random_parameters(TdokN)
    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMM(T=T,
                d=d,
                k=k,
                o=o,
                n_restarts=0,
                em_max_iter=2,
                method='mean_field',
                verbose=False)

    fhmm.fit(X)


@pytest.mark.slow
@settings(max_examples=1, deadline=None)
@given(arrays(np.int32, 5, elements=st.integers(1, 3)))
def test_gibbs_fit_runs(TdokN):
    T, d, o, k, N = adjust_random_parameters(TdokN)
    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMM(T=T,
                d=d,
                k=k,
                o=o,
                n_restarts=0,
                em_max_iter=2,
                method='gibbs',
                verbose=False)

    fhmm.fit(X)


@pytest.mark.slow
@settings(max_examples=1, deadline=None)
@given(arrays(np.int32, 5, elements=st.integers(1, 3)))
def test_sva_fit_runs(TdokN):
    T, d, o, k, N = adjust_random_parameters(TdokN)
    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMM(T=T,
                d=d,
                k=k,
                o=o,
                n_restarts=0,
                em_max_iter=2,
                method='sva',
                verbose=False)

    fhmm.fit(X)


@pytest.mark.slow
def test_iteration_saving_without_x(tmp_path):
    T=100
    d=2
    k=2
    o=1
    N=1
    n_restarts = 3
    em_max_iter = 3

    np.random.seed(2)

    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMM(T=T,
                d=d,
                k=k,
                o=o,
                n_restarts=n_restarts,
                em_max_iter=em_max_iter,
                verbose=False,
                save_directory=tmp_path,
                save_x=False)

    fhmm.fit(X)

    tstamp = os.listdir(tmp_path)[-1]

    files = []
    folders = []
    for f in os.listdir(os.path.join(tmp_path, tstamp)):
        if os.path.isfile(os.path.join(tmp_path, tstamp, f)):
            files.append(f)
        else:
            folders.append(f)

    restart_ids = folders
    assert len(restart_ids) == n_restarts + 1
    assert len(files) == 0

    for restart_id in restart_ids:
        restart_dir = os.path.join(tmp_path, tstamp, restart_id)
        pkl_files = os.listdir(restart_dir)

        assert len(pkl_files) == em_max_iter

@pytest.mark.slow
def test_iteration_saving_with_x(tmp_path):
    T=100
    d=2
    k=2
    o=1
    N=1
    n_restarts = 3
    em_max_iter = 3

    np.random.seed(2)

    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMM(T=T,
                d=d,
                k=k,
                o=o,
                n_restarts=n_restarts,
                em_max_iter=em_max_iter,
                verbose=False,
                save_directory=tmp_path,
                save_x=True)

    fhmm.fit(X)

    tstamp = os.listdir(tmp_path)[-1]

    files = []
    folders = []
    for f in os.listdir(os.path.join(tmp_path, tstamp)):
        if os.path.isfile(os.path.join(tmp_path, tstamp, f)):
            files.append(f)
        else:
            folders.append(f)

    restart_ids = folders
    assert len(restart_ids) == n_restarts + 1
    assert len(files) == 1
    assert files[0] == "X.pkl"

    for restart_id in restart_ids:
        restart_dir = os.path.join(tmp_path, tstamp, restart_id)
        pkl_files = os.listdir(restart_dir)

        assert len(pkl_files) == em_max_iter


@pytest.mark.slow
def test_iteration_saving_new_instance_matches_old(tmp_path):
    T=100
    d=2
    k=2
    o=1
    N=1

    np.random.seed(2)

    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMM(T=T,
                d=d,
                k=k,
                o=o,
                n_restarts=0,
                em_max_iter=1,
                verbose=False,
                save_directory=tmp_path,
                save_x=True)

    fhmm.fit(X)

    dirs = [x[0] for x in os.walk(tmp_path)]
    pkl_location = os.path.join(dirs[-1], "it0.pkl")
    new_fhmm = FHMM.load_model_from_file(pkl_location, load_x_from_snapshot=True)

    fhmm_params = fhmm.get_params()
    new_fhmm_params = new_fhmm.get_params()

    np.testing.assert_allclose(fhmm.X, new_fhmm.X)
    np.testing.assert_allclose(fhmm.s_exps, new_fhmm.s_exps)
    np.testing.assert_allclose(fhmm.ss_exps, new_fhmm.ss_exps)
    np.testing.assert_allclose(fhmm.sstm1_exps, new_fhmm.sstm1_exps)
    np.testing.assert_allclose(fhmm.W, new_fhmm.W)
    np.testing.assert_allclose(fhmm.A, new_fhmm.A)
    np.testing.assert_allclose(fhmm.C, new_fhmm.C)
    np.testing.assert_allclose(fhmm.pi, new_fhmm.pi)

    for key in fhmm_params:
        if "_fixed" not in key:
            assert fhmm_params[key] == new_fhmm_params[key]

@pytest.mark.slow
def test_iteration_loading_new_instance_throws_error(tmp_path):
    T=100
    d=2
    k=2
    o=1
    N=1

    np.random.seed(2)

    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMM(T=T,
                d=d,
                k=k,
                o=o,
                n_restarts=0,
                em_max_iter=1,
                verbose=False,
                save_directory=tmp_path,
                save_x=False)

    fhmm.fit(X)

    dirs = [x[0] for x in os.walk(tmp_path)]
    pkl_location = os.path.join(dirs[-1], "it0.pkl")

    with pytest.raises(AssertionError):
        new_fhmm = FHMM.load_model_from_file(pkl_location, load_x_from_snapshot=True)

@pytest.mark.slow
def test_iteration_loading_load_x_from_elsewhere(tmp_path):
    T=100
    d=2
    k=2
    o=1
    N=1

    np.random.seed(2)

    X, _ = get_random_data(T, d, k, o, N)
    save_path = os.path.join(tmp_path,'X.pkl')

    with open(save_path, 'wb') as f:
        pickle.dump(X, f)

    fhmm = FHMM(T=T,
                d=d,
                k=k,
                o=o,
                n_restarts=0,
                em_max_iter=1,
                verbose=False,
                save_directory=tmp_path,
                save_x=False)

    fhmm.fit(X)

    dirs = [x[0] for x in os.walk(tmp_path)]
    pkl_location = os.path.join(dirs[-1], "it0.pkl")

    new_fhmm = FHMM.load_model_from_file(pkl_location, x_filepath=save_path)

    np.testing.assert_allclose(X, new_fhmm.X)

@pytest.mark.slow
def test_load_best_models_from_directories(tmp_path):
    T1=100
    d1=2
    k1=2
    o1=1
    N=1
    n_restarts1 = 3
    em_max_iter1 = 3
    T2=500
    d2=3
    k2=1
    o2=2
    n_restarts2 = 0
    em_max_iter2 = 1

    np.random.seed(2)

    X1, _ = get_random_data(T1, d1, k1, o1, N)
    X2, _ = get_random_data(T2, d2, k2, o2, N)

    fhmm1 = FHMM(T=T1,
                 d=d1,
                 k=k1,
                 o=o1,
                 n_restarts=n_restarts1,
                 em_max_iter=em_max_iter1,
                 verbose=False,
                 save_directory=tmp_path,
                 save_x=True)
    fhmm2 = FHMM(T=T2,
                 d=d2,
                 k=k2,
                 o=o2,
                 n_restarts=n_restarts2,
                 em_max_iter=em_max_iter2,
                 verbose=False,
                 save_directory=tmp_path,
                 save_x=True)

    fhmm1.fit(X1)
    time.sleep(1) # Ensures timestamps are different
    fhmm2.fit(X2)

    save_dirs = [os.path.join(tmp_path, d) for d in os.listdir(tmp_path)]
    reloaded_fhmms = FHMM.load_best_models_from_directories(save_dirs)

    if reloaded_fhmms[0].d == d1:
        reloaded_fhmm1 = reloaded_fhmms[0]
        reloaded_fhmm2 = reloaded_fhmms[1]
    else:
        reloaded_fhmm2 = reloaded_fhmms[0]
        reloaded_fhmm1 = reloaded_fhmms[1]

    for fhmm, reloaded_fhmm in zip([fhmm1, fhmm2],
                                   [reloaded_fhmm1, reloaded_fhmm2]):
        np.testing.assert_allclose(fhmm.W, reloaded_fhmm.W)
        np.testing.assert_allclose(fhmm.A, reloaded_fhmm.A)
        np.testing.assert_allclose(fhmm.C, reloaded_fhmm.C)
        np.testing.assert_allclose(fhmm.pi, reloaded_fhmm.pi)


@pytest.mark.slow
def test_hessian_runs():
    T = 10
    d = 2
    o = 1
    k = 2
    N = 1

    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMM(T=T, o=o, k=k, n_restarts=0, em_max_iter=1)
    fhmm.fit(X)

    hess = fhmm.hessian()

@pytest.mark.slow
def test_cv_fits(tmp_path):
    T=100
    d=2
    k=2
    o=1
    N=1
    n_restarts = 3
    em_max_iter = 3
    n_splits = 5

    np.random.seed(2)

    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMMCV(T=T,
                  d=d,
                  k=k,
                  o=o,
                  n_restarts=n_restarts,
                  em_max_iter=em_max_iter,
                  verbose=False,
                  save_directory=tmp_path,
                  n_splits=n_splits)

    fhmm.fit(X)

    root_dir = os.listdir(tmp_path)[0]
    cv_dirs = sorted(os.listdir(os.path.join(tmp_path, root_dir)))

    for i in range(n_splits):
        cv_dir = cv_dirs[i]
        assert cv_dir == f"time_split_{i}"

        restart_dirs = os.listdir(os.path.join(tmp_path, root_dir, cv_dir))
        assert len(restart_dirs) == n_restarts + 1

        for restart_dir in restart_dirs:
            iteration_dirs = os.listdir(os.path.join(tmp_path, root_dir, cv_dir, restart_dir))
            assert len(iteration_dirs) <= em_max_iter

@pytest.mark.slow
def test_load_models_for_all_splits(tmp_path):
    T=100
    d=2
    k=2
    o=1
    N=1
    n_restarts = 3
    em_max_iter = 3
    n_splits = 3
    test_size=0.4
    subsequence_size=0.25

    np.random.seed(2)

    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMMCV(T=T,
                  d=d,
                  k=k,
                  o=o,
                  n_restarts=n_restarts,
                  em_max_iter=em_max_iter,
                  verbose=False,
                  save_directory=tmp_path,
                  test_size=test_size,
                  subsequence_size=subsequence_size,
                  n_splits=n_splits)

    fhmm.fit(X)

    save_dir = os.path.join(tmp_path, os.listdir(tmp_path)[0])
    loaded_models = FHMMCV.load_models_for_all_splits(save_dir)
    expected_T = T * subsequence_size * (1-test_size) + 1

    assert len(loaded_models) == n_splits
    for model in loaded_models.values():
        assert model.T == expected_T
        assert model.d == d
        assert model.o == o
        assert model.k == k

@pytest.mark.slow
def test_bootstrap_fits(tmp_path):
    T=100
    d=2
    k=2
    o=1
    N=1
    n_restarts = 3
    em_max_iter = 3
    sample_size = 50
    n_bootstrap_samples = 5

    np.random.seed(2)

    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMMBS(T=T,
                  d=d,
                  k=k,
                  o=o,
                  n_restarts=n_restarts,
                  em_max_iter=em_max_iter,
                  verbose=False,
                  save_directory=tmp_path,
                  sample_size=sample_size,
                  n_bootstrap_samples=n_bootstrap_samples)

    fhmm.fit(X)

    root_dir = os.listdir(tmp_path)[0]
    bootstrap_dirs = sorted(os.listdir(os.path.join(tmp_path, root_dir)))

    for i in range(n_bootstrap_samples):
        bootstrap_dir = bootstrap_dirs[i]
        assert bootstrap_dir == f"bootstrap_sample_{i}"

        restart_dirs = os.listdir(os.path.join(tmp_path, root_dir, bootstrap_dir))
        assert len(restart_dirs) == n_restarts + 1

        for restart_dir in restart_dirs:
            iteration_dirs = os.listdir(os.path.join(tmp_path, root_dir, bootstrap_dir, restart_dir))
            assert len(iteration_dirs) <= em_max_iter

@pytest.mark.slow
def test_bootstrap_sample_size_too_small(tmp_path):
    T=100
    d=2
    k=2
    o=1
    N=1
    n_restarts = 3
    em_max_iter = 3
    sample_size = -10
    n_bootstrap_samples = 5

    np.random.seed(2)

    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMMBS(T=T,
                  d=d,
                  k=k,
                  o=o,
                  n_restarts=n_restarts,
                  em_max_iter=em_max_iter,
                  verbose=False,
                  save_directory=tmp_path,
                  sample_size=sample_size,
                  n_bootstrap_samples=n_bootstrap_samples)

    with pytest.raises(AssertionError):
        fhmm.fit(X)

@pytest.mark.slow
def test_bootstrap_sample_size_too_large(tmp_path):
    T=100
    d=2
    k=2
    o=1
    N=1
    n_restarts = 3
    em_max_iter = 3
    sample_size = 101
    n_bootstrap_samples = 5

    np.random.seed(2)

    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMMBS(T=T,
                  d=d,
                  k=k,
                  o=o,
                  n_restarts=n_restarts,
                  em_max_iter=em_max_iter,
                  verbose=False,
                  save_directory=tmp_path,
                  sample_size=sample_size,
                  n_bootstrap_samples=n_bootstrap_samples)

    with pytest.raises(AssertionError):
        fhmm.fit(X)

@pytest.mark.slow
def test_load_models_for_all_bootstraps(tmp_path):
    T=100
    d=2
    k=2
    o=1
    N=1
    n_restarts = 3
    em_max_iter = 3
    sample_size = 50
    n_bootstrap_samples = 5

    np.random.seed(2)

    X, _ = get_random_data(T, d, k, o, N)

    fhmm = FHMMBS(T=T,
                  d=d,
                  k=k,
                  o=o,
                  n_restarts=n_restarts,
                  em_max_iter=em_max_iter,
                  verbose=False,
                  save_directory=tmp_path,
                  sample_size=sample_size,
                  n_bootstrap_samples=n_bootstrap_samples)

    fhmm.fit(X)

    save_dir = os.path.join(tmp_path, os.listdir(tmp_path)[0])
    loaded_models = FHMMBS.load_models_for_all_bootstraps(save_dir)

    assert len(loaded_models) == n_bootstrap_samples
    for model in loaded_models.values():
        assert model.T == sample_size
        assert model.d == d
        assert model.o == o
        assert model.k == k
