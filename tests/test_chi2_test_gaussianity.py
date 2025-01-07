import numpy as np

from nomopy.hos import chi2_test_gaussianity


def test_chi2_test_gaussianity():
    timeseries = np.random.normal(0, 1, size=100)
    dt, segment_length, f_l, f_h = 1, 10, 0.1, 0.3
    chi2, chi2_alpha = chi2_test_gaussianity(timeseries,
                                             dt,
                                             f_l,
                                             f_h,
                                             n_segments=10,
                                             alpha=0.05)

    reject_null = chi2 > chi2_alpha

    assert not reject_null
