#!/usr/bin/env python3
"""BayesianOptimization"""


import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """BayesianOptimization"""
    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """bayesian optimization"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0],
                               bounds[1], num=ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.X = X_init
        self.Y = Y_init

    def acquisition(self):
        """acquisition function"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is True:
            Y_sample_opt = np.min(self.Y)
            imp = Y_sample_opt - mu - self.xsi
        else:
            Y_sample_opt = np.max(self.Y)
            imp = mu - Y_sample_opt - self.xsi
        with np.errstate(divide='warn'):
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return self.X_s[np.argmax(EI)], EI
    
    def optimize(self, iterations=100):
        """optimize"""
        for _ in range(iterations):
            X_new, _ = self.acquisition()
            Y_new = self.f(X_new)
            self.gp.update(X_new, Y_new)
            self.X = np.vstack((self.X, X_new))
            self.Y = np.vstack((self.Y, Y_new))
        if self.minimize is True:
            idx = np.argmin(self.Y)
        else:
            idx = np.argmax(self.Y)
        return self.X[idx], self.Y[idx]
