""" Compare Bayesian models with the Bayes Factor

Examples
--------
>>> from numpy import arange, randn
>>> import models
>>> import model_selection

>>> X = randn(4000)
>>> Y = 2*X - 1 + randn(4000) # data actually is linear
>>> m1 = models.linear(X=X, Y=Y, order=1)
>>> m2 = models.linear(X=X, Y=Y, order=2)
>>> K = model_selection.bayes_factor(m1, m2)
>>> assert K >= 1
>>> assert K < 3

>>> Y = 2*X**2 - 1 + randn(1000) # data actually is quadratic
>>> m1 = models.linear(X=X, Y=Y, order=1)
>>> m2 = models.linear(X=X, Y=Y, order=2)
>>> K = model_selection.bayes_factor(m1, m2)
>>> assert K >= 100

>>> import data
>>> m1=models.linear(X=data.hdi, Y=data.tfr, order=2)
>>> m2=models.piecewise_linear(X=data.hdi, Y=data.tfr)
>>> model_selection.bayes_factor(m1, m2)
"""


from pymc import *
from numpy import exp, mean, log, array


def bayes_factor(m1, m2, iter=1e6, burn=25000, thin=10, verbose=0):
    """ Approximate the Bayes factor as the harmonic mean posterior liklihood::
        K = Pr[data | m2] / Pr[data | m1]
          ~= 1/mean(1/Pr[data_i, theta_i])

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
    m1 : object containing PyMC model vars and logp method
    m2 : object containing PyMC model vars and logp method
      m1 and m2 must each include a method called 'logp', which
      calculates the posterior log probability at all MCMC samples
      stored in the stochastic traces
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
    MCMC(m1).sample(iter*thin+burn, burn, thin, verbose=verbose)
    logp1 = m1.logp()

    MCMC(m2).sample(iter*thin+burn, burn, thin, verbose=verbose)
    logp2 = m2.logp()

    # pymc.flib.logsum suggested by Anand Patil, http://gist.github.com/179657
    K = exp(pymc.flib.logsum(-logp1) - log(len(logp1))
            - (pymc.flib.logsum(-logp2) - log(len(logp2))))

    return K


class binomial_model:
    def __init__(self, n=115, N=200, q=.5):
        self.n = n
        self.N = N
        self.q = q

        @potential
        def data_potential(n=self.n, N=self.N,
                           q=self.q):
            return binomial_like(n, N, q)
        self.data_potential = data_potential

    def logp(self):
        from numpy import array

        if isinstance(self.q, float):
            return array([Model(self).logp])

        elif isinstance(self.q, Variable) and isinstance(self.q.trace, bool):        # trace is not a function until MCMC is run
            return array([Model(self).logp])

        logp = []
        for q_val in self.q.trace():
            self.q.value = q_val
            logp.append(Model(self).logp)
        return array(logp)


def wikipedia_example(iter=1e6, burn=25000, thin=10, verbose=1):
    """ Based on the Wikipedia example, 115 heads from 200 coin
    tosses, m1 = fair coin, m2 = coin with probability q, uniform
    [0,1] prior on q.
    
    >>> assert wikipedia_example() > 1.196
    >>> assert wikipedia_example() < 1.198
    """
    m1 = binomial_model(115, 200, .5)
    m2 = binomial_model(115, 200, Uniform('q', 0., 1.))

    return bayes_factor(m1, m2, iter=iter, burn=burn, thin=thin, verbose=verbose)
