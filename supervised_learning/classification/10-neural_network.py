#!/usr/bin/env python3
"""neural network with one hidden layer"""
import numpy as np


class NeuralNetwork:
    """neural network with one hidden layer"""
    def __init__(self, nx, nodes):
        """constructor for class NeuralNetwork"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(nodes).reshape(1, nodes)
        self.b2 = 0
        self.A2 = 0

    def forward_prop(self, X):
        """forward propagation function"""
        Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = 1 / (1 + np.exp(-Z2))
        return self.A1, self.A2
