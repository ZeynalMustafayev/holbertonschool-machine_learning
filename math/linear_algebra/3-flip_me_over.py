#!/usr/bin/env python3
"""The shape should be returned as a list of integers"""


def matrix_transpose(matrix):
    """Function that returns the transpose of a 2D matrix"""
    transposed = []
    for i in range(len(matrix[0])):
        new_row = []
        for j in range(len(matrix)):
            new_row.append(matrix[j][i])
        transposed.append(new_row)
    return transposed
