#!/usr/bin/env python3
""" Task 10: 10. NeuralNetwork Forward Propagation """
import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer for performing
    binary classification.

    Attributes:
    W1 : numpy.ndarray
        The weight matrix for the hidden layer. Shape is (nodes, nx).
    b1 : numpy.ndarray
        The bias vector for the hidden layer. Shape is (nodes, 1).
    A1 : float
        The activated output of the hidden layer.
    W2 : numpy.ndarray
        The weight matrix for the output neuron. Shape is (1, nodes).
    b2 : float
        The bias term for the output neuron. Initialized to 0.
    A2 : float
        The activated output of the output neuron, representing the final
        prediction of the network.

    Methods:
    __init__(self, nx, nodes)
        Initializes the neural network with given input features and nodes in
        the hidden layer.
    def forward_prop(self, X):
        Computes the forward propagation of the neural network.
    """

    def __init__(self, nx, nodes):
        """
        Initializes the neural network.

        Parameters:
        nx : int
            The number of input features to the neural network.
        nodes : int
            The number of nodes in the hidden layer.

        Raises:
        TypeError
            If `nx` is not an integer or `nodes` is not an integer.
        ValueError
            If `nx` is less than 1 or `nodes` is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Retrieves the weight matrix for the hidden layer.

        Returns:
        numpy.ndarray
            The weight matrix for the hidden layer.
        """
        return self.__W1

    @property
    def b1(self):
        """
        Retrieves the bias vector for the hidden layer.

        Returns:
        numpy.ndarray
            The bias vector for the hidden layer.
        """
        return self.__b1

    @property
    def A1(self):
        """
        Retrieves the activated output of the hidden layer.

        Returns:
        float
            The activated output of the hidden layer.
        """
        return self.__A1

    @property
    def W2(self):
        """
        Retrieves the weight matrix for the output neuron.

        Returns:
        numpy.ndarray
            The weight matrix for the output neuron.
        """
        return self.__W2

    @property
    def b2(self):
        """
        Retrieves the bias term for the output neuron.

        Returns:
        float
            The bias term for the output neuron.
        """
        return self.__b2

    @property
    def A2(self):
        """
        Retrieves the activated output of the output neuron.

        Returns:
        float
            The activated output of the output neuron.
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Computes the forward propagation of the neural network.
        """
        C1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-C1))
        C2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-C2))
        return self.__A1, self.__A2