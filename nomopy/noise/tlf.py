import numpy as np
import scipy.linalg as la

from ..fhmm import FHMM


kB = 86.1733034 # Boltzmann constant [µeV]/[K]


class ThermalTLFModel:
    '''Model for a thermally-activated two-level fluctuator (TLF)

    Parameters
    ----------
    d : int
        Number of TLFs.
    sigma_white_noise : float, R+
        Std deviation of white noise background [noise unit].
    dt : float, R+
        Sampling timestep [s]
    '''
    def __init__(self, d, sigma_white_noise, dt=1.0):
        self.d = d
        self.sigma_white_noise = sigma_white_noise
        self.dt = dt
        self.parameters = None
        self.rates = None

    def set_rates(self, Ebs, ΔEs, f0=1e9, T=0.12):
        ''' Set the class attribute list of rates.

        Parameters
        ----------
        Ebs : list, length d
            Array of barrier heights [µeV], one for each TLF.
        ΔEs : list, length d
            List of detuning biases [µeV], one for each TLF.
        f0 : float, R+
            Bare attempt rate [Hz].
        T : float, R+
            Temperature [K].
        '''
        self.f0 = f0
        self.T = T

        rates = []
        for i in range(self.d):
            f_exc, f_rel = self.calculate_thermal_rates(f0, ΔEs[i], Ebs[i], T)
            rates += [[f_exc, f_rel]]
        self.rates = rates

    def generate(self, raw, time_steps=10000, n_samples=1, random_seed=1):
        r''' Generate an instance of the TLF timeseries.

        Parameters
        ----------
        raw : np.array, shape=(d, o, k)
            Array of fluctuator weights [noise unit].
        time_steps: int
            Length of the time series as a number of discrete samples at the
            previously specified sample rate.
        n_samples : int
            Number of independent instances of the timeseries to generate.
        random_seed : int
            Set the random seed for reproducibility.

        Returns
        _______
        t: np.array, shape=(sequence length,)
            Sampled time values.
        X: np.array, shape=(samples, sequence length, observable dim)
            Time series array.
        '''

        self._check_rates()

        k, o = 2, 1
        parameters = {'T': time_steps,
                      'd': self.d,
                      'k': 2,
                      'o': 1,
                      }
        self.time_steps = time_steps
        self.parameters = parameters.copy()

        ### Define the transition matrices: ###
        self.A = np.zeros(shape=(parameters['d'], parameters['k'], parameters['k']))
        for i in range(parameters['d']):
            self.A[i] = self.build_transition_matrix(self.rates[i][0],
                                                     self.rates[i][1],
                                                     self.dt)
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.A = np.log(self.A)

        ### Define the fluctuator weights: ###
        self.W = np.zeros(shape=(parameters['d'], parameters['o'], parameters['k']))
        for i in range(parameters['d']):
            self.W[i] = raw[i]

        ### Define the white noise background: ###
        self.C = self.sigma_white_noise**2 * np.eye(parameters['o'])

        ### Define a random initial state: ###
        np.random.seed(random_seed)
        self.pi = np.random.uniform(1E-9, 1, size=parameters['d']*parameters['k'])\
                      .reshape(parameters['d'], parameters['k'])
        for i in range(parameters['d']):
            self.pi[i] = np.array([1, 0])

        ### Generate the timeseries: ###
        fhmm = FHMM(**parameters,
                    W_fixed=self.W,
                    A_fixed=self.A,
                    C_fixed=self.C,
                    pi_fixed=self.pi)
        np.random.seed(random_seed)
        X, states = fhmm.generate(n_samples, time_steps, return_states=True)
        self.X = X
        self.states = states

        # Sampled time values
        self.t = np.linspace(self.dt, self.dt*time_steps, time_steps)

        return self.t, X

    def _check_rates(self):
        if self.rates is None:
            raise ValueError("Must set rates first via `set_rates()`")

    @staticmethod
    def build_rate_matrix(f01, f10):
        r''' Continuous-time generator of two-state Markov process.

        Parameters
        ----------
        f01 : Transition rate from 0 to 1 [1/time]
        f10 : Transition rate from 1 to 0 [1/time]

        Returns
        -------
        M : Generator of continuous time Markov process \dot{s} = M.s
        '''
        M = np.array([[-f01,  f10],
                      [ f01, -f10]])
        return M

    @classmethod
    def build_transition_matrix(cls, f01, f10, dt):
        r''' Transition matrix for the discrete-time Markov process.

        Parameters
        ----------
        f01 : float, [0, 1]
            Transition rate from 0 to 1 [1/time]
        f10 : float, [0, 1]
            Transition rate from 1 to 0 [1/time]
        dt : float, R+
            Sampling time interval [time]

        Returns
        -------
        P: Probability transition matrix
        '''
        P = la.expm(cls.build_rate_matrix(f01, f10) * dt)
        return P

    @staticmethod
    def calculate_thermal_rates(f0, ΔE, Eb, T):
        r''' Thermal rates.

        Parameters
        ----------
        f0 : float, R+
            Bare attempt rate [Hz].
        ΔEs : list, length d
            List of detuning biases [µeV], one for each TLF.
        Ebs : list, length d
            Array of barrier heights [µeV], one for each TLF.
        T : float, R+
            Temperature [K].

        Returns
        -------
        freq_excitation, freq_relaxation : tuple
            Excitation frequency and relaxation frequency.

        '''
        # In zero-temperature limit, dynamics is frozen
        if T == 0:
            return 0, 0

        excitation = (Eb - 0.5*ΔE) / (kB*T)
        relaxation = (Eb + 0.5*ΔE) / (kB*T)

        freq_excitation = f0 * np.exp(-excitation) if excitation <= 100 else 0
        freq_relaxation = f0 * np.exp(-relaxation) if relaxation <= 100 else 0

        return freq_excitation, freq_relaxation

    @staticmethod
    def calculate_tlf_psd(frequency,
                          weight,
                          γ0,
                          γ1):
        r'''Calculates the analytic Lorentzian PSD of a TLF.

        Parameters
        ----------
        frequency : float, np.array
            Frequencies at which to evaluate the PSD.

        weight : 

        γ0 : 

        γ1 : 

        Returns
        -------
        S : float, np.array
            One-sided PSD, S(frequency), in terms of frequency.
        '''
        rate_sum = γ0 + γ1
        sigma, tau = 1/γ0, 1/γ1
        prefactor = weight**2 * 1/np.pi * rate_sum * sigma * tau / (sigma + tau)**2
        S = (4*np.pi) * prefactor / ((2*np.pi*frequency)**2 + rate_sum**2)
        return S
