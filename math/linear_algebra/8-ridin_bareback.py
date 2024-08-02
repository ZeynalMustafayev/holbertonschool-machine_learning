#!/usr/bin/env python3
"""Module that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Function that performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    new_matrix = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            mul = 0
            for z in range(len(mat2)):
                mul += mat1[i][z] * mat2[z][j]
            row.append(mul)
        new_matrix.append(row)
    return new_matrix
