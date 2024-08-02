#!/usr/bin/env python3
"""Module that concatenates two 2D matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Function that concatenates two 2D matrices"""
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    if axis == 1 and len(mat1) != len(mat2):
        return None
    new_matrix = []
    if axis == 0:
        for i in mat1:
            new_matrix.append(i.copy())
        for j in mat2:
            new_matrix.append(j.copy())
    if axis == 1:
        for i in range(len(mat1)):
            new_matrix.append(mat1[i] + mat2[i])
    return new_matrix
