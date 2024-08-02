#!/usr/bin/env python3
"""Module that adds two 2D matrices"""


def add_matrices2D(mat1, mat2):
    """Function that adds two 2D matrices"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    matrix = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1)):
            sum = mat1[i][j] + mat2[i][j]
            row.append(sum)
        matrix.append(row)
    return matrix
