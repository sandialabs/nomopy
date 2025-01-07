import numpy as np
import pytest

from nomopy.noise import ThermalTLFModel


def test_build_rate_matrix():
    exp = np.array([[-1, 1], [1, -1]])
    act = ThermalTLFModel.build_rate_matrix(1, 1)
    assert np.all(exp == act)


def test_build_transition_matrix():
    f01, f10, dt = 1, 1, 1
    P = ThermalTLFModel.build_transition_matrix(f01, f10, dt)
    assert P.shape[0] == 2 and P.shape[1] == 2
    assert np.allclose(P.sum(axis=0), np.array([1, 1]))


def test_calculate_thermal_rates_zero_T():
    act_f_exc, act_f_rel = ThermalTLFModel.calculate_thermal_rates(1, 1, 1, 1)

    assert act_f_exc > 0
    assert act_f_rel > 0


def test_calculate_thermal_rates_nonzero_T():
    exp_f_exc = 0
    exp_f_rel = 0

    act_f_exc, act_f_rel = ThermalTLFModel.calculate_thermal_rates(1, 1, 1, 0)

    assert exp_f_exc == act_f_exc
    assert exp_f_rel == act_f_rel


def test_calculate_tlf_psd():
    S = ThermalTLFModel.calculate_tlf_psd(1, 1, 1, 1)
    assert float(S)


def test_calculate_tlf_psd_array():
    f = np.array([1, 2, 3])
    S = ThermalTLFModel.calculate_tlf_psd(f, 1, 1, 1)

    assert len(S) == 3


def test_generate_without_set_rates_raises():
    d = 2
    o = 1
    k = 2
    σ = 1  # white noise
    tlfs = ThermalTLFModel(d, σ)
    raw = np.random.rand(d*o*k).reshape(d, o, k)

    with pytest.raises(ValueError):
        rat, X = tlfs.generate(raw, time_steps=10, n_samples=1, random_seed=1)


def test_generate():
    d = 2
    o = 1
    k = 2
    σ = 1  # white noise
    tlfs = ThermalTLFModel(d, σ)
    raw = np.random.rand(d*o*k).reshape(d, o, k)
    tlfs.set_rates([0.1, 0.1], [0.1, 0.1])
    rat, X = tlfs.generate(raw, time_steps=10, n_samples=1, random_seed=1)

    assert len(rat) == 10
    assert X.shape[0] == 1  # num samples
    assert X.shape[1] == 10 # T
    assert X.shape[2] == 1  # o


def test_set_rates():
    d = 2
    o = 1
    k = 2
    σ = 1  # white noise
    tlfs = ThermalTLFModel(d, σ)
    raw = np.random.rand(d*o*k).reshape(d, o, k)
    tlfs.set_rates([0.1, 0.1], [0.1, 0.1])

    assert len(tlfs.rates) == d
    for tlf_idx in range(d):
        assert len(tlfs.rates[tlf_idx]) == k
