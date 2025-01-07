Modules 
*******

Summary of modules of nomopy

* Factorial hidden Markov models (nomopy.fhmm):
   * FHMM: Implementation of a factorial hidden Markov model, including (1) several expectation-maximization (EM) algorithms
     for model parameter fitting, (2) confidence interval generation, (3) hidden state trajectory reconstruction, and (4)
     data generation
   * FHMMCV: Machinery for cross-validation and model selection through train-test splitting of noise timeseries
   * Hessian: Machinery for computing the Hessian of the loglikelihood with respect to model parameters
* Higher order statistics (nomopy.hos):
   * second_spectrum(): Evaluation of the second spectrum, as defined in [SeidlerSolin1996]_
   * chi2_test_gaussianity(): chi-squared test for Gaussianity of the noise timeseries, based on the second spectrum
* Noise models (nomopy.noise):
   * ThermalTLFModel: Model for an ensemble of thermally-activated two-level fluctuators

.. [SeidlerSolin1996] G.T. Seidler and S.A. Solin, Physical Review B 53, 9753 (1996)


nomopy.fhmm.FHMM
================

Implementation of a factorial hidden Markov model, including (1) several expectation-maximization (EM) algorithms for model parameter fitting, (2) confidence interval generation, (3) hidden state trajectory reconstruction, and (4) data generation

.. autoclass:: nomopy.fhmm.FHMM.FHMM
   :members: canonically_transform_W,
             expected_complete_log_likelihood_sample,
             expected_complete_log_likelihood,
             fit,
             generate,
             generate_random_model_params,
             hessian,
             kld,
             log_likelihood_sample,
             log_likelihood,
             plot_fit,
             standard_errors,
             viterbi

nomopy.fhmm.FHMMCV
==================

Machinery for cross-validation and model selection through train-test splitting of noise timeseries

.. autoclass:: nomopy.fhmm.FHMM.FHMMCV
   :members: time_splits, fit

nomopy.fhmm.Hessian
===================

.. automodule:: nomopy.fhmm.exact_hessian

.. autoclass:: nomopy.fhmm.exact_hessian.Hessian
   :members: hessian_element, hessian

nomopy.hos
==========

.. automodule:: nomopy.hos
   :members:

.. automethod:: nomopy.hos.second_spectrum

.. automethod:: nomopy.hos.chi2_test_gaussianity

nomopy.noise
============
.. automodule:: nomopy.noise
   :members:

.. autoclass:: nomopy.noise.tlf.ThermalTLFModel
   :members:
