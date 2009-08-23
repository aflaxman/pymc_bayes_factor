""" Compare models for TFR ~ f(HDI) with Bayes Factor

Examples
--------
>>> from numpy import arange, randn
>>> import models
>>> import model_selection

>>> X = randn(1000)
>>> Y = 2*X - 1 + randn(1000) # data actually is linear
>>> m1 = models.linear(X=X, Y=Y, order=1)
>>> m2 = models.linear(X=X, Y=Y, order=2)
>>> K = model_selection.bayes_factor(m1, m2)
>>> assert K > 1
>>> assert K < 3

>>> Y = 2*X**2 - 1 + randn(1000) # data actually is quadratic
>>> m1 = models.linear(X=X, Y=Y, order=1)
>>> m2 = models.linear(X=X, Y=Y, order=2)
>>> K = model_selection.bayes_factor(m1, m2)
>>> assert K > 100

>>> import data
>>> m1=models.linear(X=data.hdi, Y=data.tfr, order=2)
>>> m2=models.piecewise_linear(X=data.hdi, Y=data.tfr)
>>> model_selection.bayes_factor(m1, m2)
"""


from pymc import *
from numpy import exp, mean

import data
import models


def bayes_factor(m1, m2, iter=1e6, burn=25000, thin=1, verbose=0):
    """ Approximate the Bayes factor::
        K = Pr[data | m2] / Pr[data | m1]

    to compare 2 models.  According to Wikipedia / Jefferies, the
    interpretation of K is the following::

        K < 1         -   data supports m1
        1 <= K < 3    -   data supports m2, but is "Barely worth mentioning"
        3 <= K < 10   -   data gives "Substantial" support for m2
        10 <= K < 30  -   "Strong" support for m2
        30 <= K < 100 -   "Very strong" support for m2
        K >= 100      -   "Decisive" support for m2

    Parameters
    ----------
    m1 : dict of PyMC model vars
    m2 : dict of PyMC model vars
      m1 and m2 must each include a key called 'data_logp', which is
      usually a PyMC deterministic that takes values equal to the log
      of the probability of the data under the current model
      parameters
    iter: int, optional
      number of iterations of MCMC
    burn: int, optional
      number of initial MCMC iters to discard
    thin: int, optional
      number of MCMC iters to discard per sample
    verbose : int, optional
      amount of output to request from PyMC

    Results
    -------
    K : estimate of the bayes factor
    """

    mc1 = MCMC(m1)
    mc1.sample(iter, burn, thin, verbose=verbose)
    
    mc2 = MCMC(m2)
    mc2.sample(iter, burn, thin, verbose=verbose)

    mu_logp = m1['data_logp'].stats()['mean']
    K = mean(exp(m2['data_logp'].trace() - mu_logp)) / mean(exp(m1['data_logp'].trace() - mu_logp))

    return K
