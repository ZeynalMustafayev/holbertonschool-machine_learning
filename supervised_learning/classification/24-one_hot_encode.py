#!/usr/bin/env python3
""" Task 24: 24. One-Hot Encode """
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot encoded
    matrix.

    Args:
        Y (numpy.ndarray): Array of shape (m,) containing
        the numeric class labels.
            - m (int): The number of examples.
        classes (int): The total number of unique classes.

    Returns:
        numpy.ndarray: A one-hot encoded matrix of shape
        (classes, m), where each column
        corresponds to a one-hot encoded vector for each
        label in `Y`.
        - If the input is invalid, returns None.
    """
    if Y is None\
       or type(Y) is not np.ndarray\
       or type(classes) is not int:
        return None
    try:
        matrix = np.zeros((len(Y), classes))
        matrix[np.arange(len(Y)), Y] = 1
        return matrix.T
    except Exception:
        return None
