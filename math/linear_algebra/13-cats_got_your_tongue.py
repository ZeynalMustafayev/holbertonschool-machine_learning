#!/usr/bin/env python3
"""Module that concatenates two numpy.ndarrays"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Function that concatenates two numpy.ndarrays"""
    return np.concatenate((mat1, mat2), axis)
