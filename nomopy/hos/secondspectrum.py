import numpy as np
from numpy.fft import rfft, irfft
from numpy.fft import rfftfreq
import scipy.stats as scs


def second_spectrum(timeseries, dt, segment_length, f_l, f_h, method=''):
    """ 'Second Spectrum' as defined in Seidler and Solin 1996
    """
    nsegments = timeseries.shape[0] // segment_length

    # Time step between each power spectrum
    T = dt * segment_length

    b_l = int(T * f_l)
    b_h = int(T * f_h)

    if not (b_h < segment_length // 2):
        raise ValueError("Must have f_h < 1/(2 * dt) for valid frequency"
                         " band index.")

    s2s = []
    s2s_gauss = []
    s1s = []
    norms = []

    for i in range(nsegments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length
        subseries = timeseries[start_idx:end_idx]

        A = rfft(subseries)
        freqs = rfftfreq(subseries.shape[0], dt)

        # Bandpass filter
        A[:b_l] = 0
        A[b_h:] = 0

        # First spectrum
        s1 = 2 * T * np.abs(A)**2
        s1s.append(s1)

        if method == 'amplitude':
            norms.append(np.sum(s1) / T)
            # Amplitude Second Spectrum -- Randomizing phase
            phase = 2 * np.pi * np.random.rand(len(A[A != 0]))
            A[A != 0] = np.abs(A[A != 0]) * np.exp(1j * phase)
        elif method == 'phase':
            norms.append(2 * T * (f_h - f_l))
            # Phase Second Spectrum -- Removing amplitude
            phase = np.angle(A[A != 0])
            A[A != 0] = np.exp(1j * phase)
        else:
            norms.append(np.sum(s1) / T)

        # Second spectrum
        s2 = np.zeros(shape=(b_h-b_l,))
        for p in range(0, b_h-b_l):
            Akp = A[b_l+p:b_h]
            Ak_star = np.conjugate(A[b_l:b_h-p])
            Anp_star = np.conjugate(A[b_l+p:b_h])
            An = A[b_l:b_h-p]
            s2[p] += (8 * T) * np.real(np.sum(Akp * Ak_star) * np.sum(Anp_star * An))

        # Second spectrum
        s2_gauss = np.zeros(shape=(b_h-b_l,))
        for p in range(0, b_h-b_l):
            Anp = A[b_l+p:b_h]
            Anp_star = np.conjugate(Anp)
            An = A[b_l:b_h-p]
            An_star = np.conjugate(An)
            s2_gauss[p] += (8 * T) * np.real(np.sum(Anp * Anp_star * An * An_star))

        s2s.append(s2)
        s2s_gauss.append(s2_gauss)

    s2s = np.array(s2s)
    norm = sum(norms) / len(norms)
    # Frequency range for the second spectrum
    freqs = np.arange(len(s2))/T

    if method == 'all':
        return s2s/norm**2, freqs

    # Average the second spectrums and normalize by the square of the
    # average integrated, bandlimited noise power
    s2_mean = s2s.mean(axis=0)
    s2_mean /= norm**2
    s2_std = s2s.std(axis=0)
    s2_std /= norm**2

    s2_gauss = sum(s2s_gauss) / len(s2s_gauss)
    norm = sum(norms) / len(norms)
    s2_gauss = s2_gauss / norm**2

    return s2_mean, s2_std, s2_gauss, freqs


def chi2_test_gaussianity(timeseries,
                          dt,
                          f_l,
                          f_h,
                          n_segments=50,
                          alpha=0.05):
    ''' Evaluate chi-squared test for "non-Gaussianity" based on the
        second spectrum.
    '''
    segment_length = int(timeseries.shape[0] // n_segments)
    s2_mean, s2_std, s2_gauss, freqs = second_spectrum(timeseries,
                                                       dt,
                                                       segment_length,
                                                       f_l,
                                                       f_h,
                                                       method='')
    s2s, freqs = second_spectrum(np.hstack([timeseries]),
                                 dt,
                                 segment_length,
                                 f_l,
                                 f_h,
                                 method='all')

    # Estimate a reasonable lambda parameter to use for a Box-Cox transformation:
    lmdas = []
    for i in range(s2s.shape[1]):
        x, lmda = scs.boxcox(s2s[:, i])
        lmdas.append(lmda)
    lmda = np.mean(lmdas)

    # Apply the Box-Cox transformation to our data:
    y = scs.boxcox(s2_mean,lmbda=lmda)
    f_y = scs.boxcox(s2_gauss,lmbda=lmda)
    σ = np.abs(s2_mean**(lmda-1) * s2_std)

    χ_i = ((y - f_y) / (σ))**2

    chi2 = np.sum(χ_i[1:])
    chi2_alpha = scs.chi2(len(s2_gauss)).ppf(1-alpha)

    # reject_null = chi2 > chi2_alpha

    return chi2, chi2_alpha
