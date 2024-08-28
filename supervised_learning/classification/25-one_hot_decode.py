#!/usr/bin/env python3
""" Task 25: 25. One-Hot Decode """
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot encoded matrix back into a
    numeric label vector.

    Args:
        one_hot (numpy.ndarray): A 2D array of shape
        (classes, m) where each column
        represents a one-hot encoded vector.

    Returns:
        numpy.ndarray: A 1D array of shape (m,)
        containing the original class labels.
        - If the input is invalid, returns None.
    """
    if type(one_hot) is not np.ndarray\
       or len(one_hot.shape) != 2:
        return None
    return np.array([np.where(i == 1)[0][0]
                     for i in one_hot.T])
