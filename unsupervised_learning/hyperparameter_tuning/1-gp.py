#!/usr/bin/env python3
"""GaussianProcess"""


import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process.
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor that initializes the Gaussian process.
        """
        # X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        # Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        # Set the covariance kernel matrix for the initial inputs
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Kernel.
        """
        x = np.sum(X1**2, axis=1).reshape(-1, 1)
        sqdist = x + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)

        # Compute the kernel using the RBF formula
        K = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

        return K

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a Gaussian proces
        """
        # Compute the covariance kernel matrix between X_s and X
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        # Compute the mean at X_s
        mu_s = K_s.T.dot(K_inv).dot(self.Y)

        # Compute the covariance at X_s
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s.reshape(-1), np.diag(cov_s)
