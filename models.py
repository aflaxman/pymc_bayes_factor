""" Functions to generate PyMC variables for linear and piecewise
linear models

Methods
-------
linear(X, Y, order=1)
  Y[i] ~ beta[0] * X[i]^order + ... + beta[order-1] * X[i] + beta[order] + N(0, sigma^2),

piecewise_linear(X, Y, breakpoint=.86)
         { beta[0] + beta[1] * (X[i]-breakpoint) + N(0, sigma^2) if X[i] < breakpoint
  Y[i] ~ {
         { beta[0] + beta[2] * (X[i]-breakpoint) + N(0, sigma^2) if X[i] >= breakpoint

Examples
--------
>>> from pymc import *
>>> import data, models

>>> m1 = models.linear(X=data.hdi, Y=data.tfr, order=2, mu_beta=[0., -8, 8], sigma_beta=5., mu_sigma=.1)
>>> MCMC(m1).sample(1000*1000 + 20000, 20000, 1000)

>>> m2 = models.piecewise_linear(X=data.hdi, Y=data.tfr, breakpoint=.86, mu_beta=[1,-8,1], sigma_beta=1., mu_sigma=.1)
>>> MCMC(m2).sample(1000*1000 + 20000, 20000, 1000)

"""


from pymc import *
from numpy import zeros, ones, array, polyval


class linear:
    def __init__(self, X, Y, order=1, mu_beta=None, sigma_beta=1., mu_sigma=1.):
        """ Make a Bayesian model for::

            Y[i] ~ beta[0] * X[i]^order + ... + beta[order-1] * X[i] + beta[order] + N(0, sigma^2),

        with priors::

            beta[i] ~ N(mu=mu_beta, tau=1/sigma_beta**2)
            sigma ~ Gamma(alpha=1, beta=1/mu_sigma)

        Parameters
        ----------
        X : list of floats
        Y : list of floats
        mu_beta : list of floats, optional
        sigma_beta : float or list of floats, optional
        mu_sigma : float, optional
        order : int, optional
          the order of the fit polynomial

        Results
        -------
        m : an object containing all relevant PyMC stochastic
          variables and also a prediction function and a log
          probability calculation function
        """
        if mu_beta == None:
            mu_beta = zeros(order+1)

        self.beta = Normal('beta', mu_beta, sigma_beta**-2)
        self.sigma = Gamma('standard error', 1., 1./mu_sigma)

        @potential
        def data_potential(beta=self.beta, sigma=self.sigma,
                           X=X, Y=Y):
            mu = self.predict(beta, X)
            return normal_like(Y, mu, 1 / sigma**2)
        self.data_potential = data_potential

    def predict(self, beta, X):
        return polyval(beta, X)

    def logp(self):
        if isinstance(self.beta.trace, bool):  # trace is not a function until MCMC is run
            return Model(self).logp
        logp = []
        for beta_val, sigma_val in zip(self.beta.trace(), self.sigma.trace()):
            self.beta.value = beta_val
            self.sigma.value = sigma_val
            logp.append(Model(self).logp)
        return array(logp)

class piecewise_linear:
    def __init__(self, X, Y, breakpoint=.86,
                 mu_beta=[0., 0., 0.], sigma_beta=1., mu_sigma=1.):
        """ Make a Bayesian model for::
                 { beta[0] + beta[1] * (X[i]-breakpoint) + N(0, sigma^2) if X[i] < breakpoint
          Y[i] ~ {
                 { beta[0] + beta[2] * (X[i]-breakpoint) + N(0, sigma^2) if X[i] >= breakpoint

        with priors::

            beta[i] ~ N(mu=mu_beta, tau=1/sigma_beta**2)
            sigma ~ Gamma(alpha=1, beta=1/mu_sigma)

        Parameters
        ----------
        X : list of floats
        Y : list of floats
        breakpoint : float or PyMC stoch
          the x value to switch from one linear model to the other
        mu_beta : list of floats, optional
        sigma_beta : float or list of floats, optional
        mu_sigma : float, optional

        Results
        -------
        m : an object containing all relevant PyMC stochastic
          variables and also a prediction function and a log
          probability calculation function
        """
        self.breakpoint=breakpoint
        self.beta = Normal('beta', mu_beta, sigma_beta**-2)
        self.sigma = Gamma('standard error', 1., 1./mu_sigma)

        @potential
        def data_potential(beta=self.beta, sigma=self.sigma,
                           X=X, Y=Y, breakpoint=self.breakpoint):
            mu = self.predict(beta, breakpoint, X)
            return normal_like(Y, mu, 1 / sigma**2)
        self.data_potential = data_potential
        
    def predict(self, beta, breakpoint, X):
            very_high_dev_indicator = X >= breakpoint
            mu = (beta[0] + beta[1]*(X-breakpoint)) * (1 - very_high_dev_indicator)
            mu += (beta[0] + beta[2]*(X-breakpoint)) * very_high_dev_indicator
            return mu

    def logp(self, beta_val=None, sigma_val=None, breakpoint_val=None):
        if isinstance(self.beta.trace, bool):  # trace is not a function until MCMC is run
            return Model(self).logp
        logp = []
        if isinstance(self.breakpoint, Variable):
            for beta_val, sigma_val, breakpoint_val in zip(self.beta.trace(), self.sigma.trace(), self.breakpoint.trace()):
                self.beta.value = beta_val
                self.sigma.value = sigma_val
                self.breakpoint.value = breakpoint_val
                logp.append(Model(self).logp)
        else:
            for beta_val, sigma_val in zip(self.beta.trace(), self.sigma.trace()):
                self.beta.value = beta_val
                self.sigma.value = sigma_val
                logp.append(Model(self).logp)
        return array(logp)
