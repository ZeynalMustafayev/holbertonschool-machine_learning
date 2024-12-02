import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process.
    """
    def __init__(self, X_init, Y_init, L=1, sigma_f=1):
        """
        Class constructor
        """
        # Inputs already sampled (t, 1)
        self.X = X_init  # Inputs
        self.Y = Y_init  # Outputs corresponding to inputs
        self.L = L       # Length parameter for the kernel
        self.sigma_f = sigma_f  # Standard deviation of output

        # Calculate the initial covariance kernel matrix
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Radial Basis Function (RBF) kernel to compute covariance matrix
        """
        # Expand dimensions to allow broadcasting
        sqdist = np.sum(X1**2,
                        axis=1).reshape(-1,
                                        1) + np.sum(X2**2,
                                                    axis=1) - 2 * np.dot(X1,
                                                                         X2.T)
        K = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
        return K
