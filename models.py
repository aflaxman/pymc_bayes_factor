""" Functions to generate PyMC variables for linear and piecewise
linear models

Methods
-------
linear(X, Y, order=1)
  Y[i] ~ beta[0] * X[i]^order + ... + beta[order-1] * X[i] + beta[order] + N(0, sigma^2),

piecewise_linear(X, Y, breakpoint=.86)
         { beta[0] + beta[1] * X[i] + N(0, sigma^2) if X[i] < breakpoint
  Y[i] ~ {
         { beta[2] + beta[3] * X[i] + N(0, sigma^2) if X[i] >= breakpoint
"""


from pymc import *
from numpy import zeros, ones, array, polyval


def linear(X, Y, order=1):
    """ Make a Bayesian model for::
    
        Y[i] ~ beta[0] * X[i]^order + ... + beta[order-1] * X[i] + beta[order] + N(0, sigma^2),
        
    with priors::

        beta[i] ~ N(mu=0, tau=10^-2)
        sigma ~ Gamma(alpha=1, beta=1)

    Parameters
    ----------
    X : list of floats
    Y : list of floats
    order : int, optional
      the order of the fit polynomial

    Results
    -------
    m : dict of PyMC variables, including a deterministic called 'data_logp'
    """
    beta = Normal('beta', zeros(order+1), 10.**-2 * ones(order+1))
    sigma = Gamma('standard error', 1., 1.)

    @deterministic
    def data_logp(beta=beta, sigma=sigma, X=X, Y=Y):
        return normal_like(Y, polyval(beta, X), 1. / sigma**2)

    @potential
    def data_potential(data_logp=data_logp):
        return data_logp

    return dict(X=X, Y=Y, beta=beta, sigma=sigma,
                data_logp=data_logp, data_potential=data_potential)

    
def piecewise_linear(X, Y, breakpoint=.86):
    """ Make a Bayesian model for::
              { beta[0] + beta[1] * X[i] + N(0, sigma^2) if X[i] < breakpoint
       Y[i] ~ {
              { beta[2] + beta[3] * X[i] + N(0, sigma^2) if X[i] >= breakpoint

    with priors::

        beta[i] ~ N(mu=0, tau=10^-2)
        sigma ~ Gamma(alpha=1, beta=1)

    Parameters
    ----------
    X : list of floats
    Y : list of floats
    breakpoint : float or stoch
      the x value to switch from one linear model to the other

    Results
    -------
    m : dict of PyMC variables, including a deterministic called 'data_logp'
    """
    beta = Normal('beta', zeros(4), 10.**-2 * ones(4))
    sigma = Gamma('standard error', 1., 1.)

    @deterministic
    def data_logp(beta=beta, sigma=sigma, X=X, Y=Y, breakpoint=breakpoint):
        very_high_dev_indicator = X >= breakpoint
        mu = (beta[0] + beta[1]*X) * (1 - very_high_dev_indicator)
        mu += (beta[2] + beta[3]*X) * very_high_dev_indicator
        return normal_like(Y, mu, 1 / sigma**2)

    @potential
    def data_potential(data_logp=data_logp):
        return data_logp

    return dict(X=X, Y=Y, breakpoint=breakpoint, beta=beta, sigma=sigma,
                data_logp=data_logp, data_potential=data_potential)
