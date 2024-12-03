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
        """
        Optimize method
        Args:
            iterations: maximum number of iterations to perform
        Returns: x_opt, y_opt
                 x_opt: numpy.ndarray of shape (1,) representing the optimal
                 point
                 y_opt: numpy.ndarray of shape (1,) representing the optimal
                 function value
        """

        X_all_s = []
        for i in range(iterations):
            # Find the next sampling point xt by optimizing the acquisition
            # function over the GP: xt = argmaxx μ(x | D1:t−1)

            x_opt, _ = self.acquisition()
            # If the next proposed point is one that has already been sampled,
            # optimization should be stopped early
            if x_opt in X_all_s:
                break

            y_opt = self.f(x_opt)

            # Add the sample to previous samples
            # D1: t = {D1: t−1, (xt, yt)} and update the GP
            self.gp.update(x_opt, y_opt)
            X_all_s.append(x_opt)

        if self.minimize is True:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        self.gp.X = self.gp.X[:-1]

        x_opt = self.gp.X[index]
        y_opt = self.gp.Y[index]

        return x_opt, y_opt
