#!/usr/bin/env python3
""" Task 19: 19. DeepNeuralNetwork Cost """
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network for performing binary classification.

    Attributes:
        nx (int): Number of input features.
        layers (list): List representing the number of nodes
        in each layer of the network.
        L (int): Number of layers in the neural network.
        cache (dict): Dictionary to hold the intermediary
        values of the network (i.e., the activations).
        weights (dict): Dictionary to hold the weights
        and biases of the network.

    Methods:
        __init__(self, nx, layers):
            Initializes the deep neural network with given input features
            and nodes in each layer.
        L(self):
            Property getter for the number of layers in the network.
        cache(self):
            Property getter for the intermediary values in the network.
        weights(self):
            Property getter for the weights and biases of the network.
        forward_prop(self, X)
            Calculates the forward propagation of the neural network.
        cost(self, Y, A)
            Calculates the cost of the model using logistic regression.
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes
            in each layer of the network.

        Raises:
            TypeError: If `nx` is not an integer.
            ValueError: If `nx` is less than 1.
            TypeError: If `layers` is not a list of positive integers.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__nx = nx
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            W_key = f"W{i + 1}"
            b_key = f"b{i + 1}"

            self.__weights[b_key] = np.zeros((layers[i], 1))

            if i == 0:
                f = np.sqrt(2 / nx)
                self.__weights[W_key] = np.random.randn(layers[i], nx) * f
            else:
                f = np.sqrt(2 / layers[i - 1])
                self.__weights[W_key] = \
                    np.random.randn(layers[i], layers[i - 1]) * f

    @property
    def L(self):
        """ Property getter for the number of
        layers in the network."""
        return self.__L

    @property
    def cache(self):
        """ Property getter for the intermediary
        values in the network. """
        return self.__cache

    @property
    def weights(self):
        """ Property getter for the weights and
        biases of the network. """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Array of shape (nx, m) containing
            the input data.
                - nx (int): Number of input features.
                - m (int): Number of examples.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The activations of the output layer
                (A_L) with shape (1, m).
                - dict: The cache dictionary containing the intermediary
                activations of each layer.
        """
        self.__cache['A0'] = X

        for i in range(self.__L):
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)
            A_key_prev = "A{}".format(i)
            A_key_forw = "A{}".format(i + 1)

            Z = np.matmul(self.__weights[W_key], self.__cache[A_key_prev]) \
                + self.__weights[b_key]
            self.__cache[A_key_forw] = 1 / (1 + np.exp(-Z))

        return self.__cache[A_key_forw], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Array with shape (1, m)
            containing the correct labels for the input data.
                - m (int): Number of examples.
            A (numpy.ndarray): Array with shape (1, m)
            containing the activated output of the network for
            each example.

        Returns:
            float: The cost of the model, calculated using the
            logistic regression cost function.
        """
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost
