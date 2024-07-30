#!/usr/bin/env python3
"""The shape should be returned as a list of integers"""


def matrix_shape(matrix):
    """Function that returns the shape of a matrix"""
    size = []
    while isinstance(matrix, list):
        size.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return size
