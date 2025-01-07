import numpy as np
import pytest

from nomopy.hos import second_spectrum


def test_second_spectrum_():
    timeseries = np.random.normal(0, 1, size=100)
    dt, segment_length, f_l, f_h = 1, 10, 0.1, 0.3
    _ = second_spectrum(timeseries,
                        dt,
                        segment_length,
                        f_l,
                        f_h)


def test_second_spectrum_all():
    timeseries = np.random.normal(0, 1, size=100)
    dt, segment_length, f_l, f_h = 1, 10, 0.1, 0.3
    _ = second_spectrum(timeseries,
                        dt,
                        segment_length,
                        f_l,
                        f_h,
                        method='all')


def test_second_spectrum_b_h_raises():
    timeseries = np.random.normal(0, 1, size=100)
    dt, segment_length, f_l = 1, 10, 0.1
    f_h = 1/(2*dt) + 1E-4  # f_h < 1/(2 * dt)

    with pytest.raises(ValueError):
        _ = second_spectrum(timeseries,
                            dt,
                            segment_length,
                            f_l,
                            f_h)


def test_second_spectrum_b_h_raises():
    timeseries = np.random.normal(0, 1, size=100)
    dt, segment_length, f_l = 1, 10, 0.1
    f_h = 1/(2*dt) - 1E-4  # f_h < 1/(2 * dt)
    _ = second_spectrum(timeseries,
                        dt,
                        segment_length,
                        f_l,
                        f_h)
