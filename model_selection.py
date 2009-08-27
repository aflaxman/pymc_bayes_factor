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


def bayes_factor(m1, m2, iter=1e6, burn=25000, thin=10, verbose=0):
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
      m1 and m2 must each include a key called 'logp', which is
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

    Notes
    -----
    This sneaky harmonic mean of liklihood values appears in Kass and
    Raftery (1995) where it is attributed to to Newton and Raftery
    (1994)
    """

    MCMC(m1).sample(iter, burn, thin, verbose=verbose)
    if verbose:  print 1. / mean(exp(-m1['data_logp'].trace()))

    MCMC(m2).sample(iter, burn, thin, verbose=verbose)
    if verbose:  print 1. / mean(exp(-m2['data_logp'].trace()))

    mu_logp = m2['data_logp'].stats()['mean']
    K = mean(exp(mu_logp-m1['data_logp'].trace())) / mean(exp(mu_logp-m2['data_logp'].trace()))

    return K

def wikipedia_example(iter=1e6, burn=25000, verbose=1):
    """ Based on the Wikipedia example, 115 heads from 200 coin
    tosses, m1 = fair coin, m2 = coin with probability q, uniform
    [0,1] prior on q.
    
    >>> assert wikipedia_example() > 1.196
    >>> assert wikipedia_example() < 1.198
    """

    # fair coin model
    @deterministic
    def data_logp():
         return binomial_like(115, 200, .5)

    @deterministic
    def logp(data_logp=data_logp):
         return data_logp

    @potential
    def data_potential(data_logp=data_logp):
        return data_logp

    m1 = dict(data_logp=data_logp, logp=logp, data_potential=data_potential)

    # coin w unknown bias model
    q = Uniform('q', 0, 1)
    @deterministic
    def data_logp(q=q):
        return binomial_like(115, 200, q)

    @deterministic
    def logp(q=q, data_logp=data_logp):
         return uniform_like(q,0,1) + data_logp

    @potential
    def data_potential(data_logp=data_logp):
        return data_logp

    m2 = dict(q=q, data_logp=data_logp, logp=logp, data_potential=data_potential)

    return m1, m2, bayes_factor(m1, m2, iter=iter, burn=burn, verbose=verbose)
