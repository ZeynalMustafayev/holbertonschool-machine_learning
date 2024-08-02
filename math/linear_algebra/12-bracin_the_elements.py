#!/usr/bin/env python3
"""Module that performs math operations on two numpy.ndarrays"""


def np_elementwise(mat1, mat2):
    """Function that performs math operations on two numpy.ndarrays"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
