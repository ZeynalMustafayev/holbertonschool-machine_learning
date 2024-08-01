#!/usr/bin/env python3
"""Module that adds two 2D matrices"""


def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2):
        return None
    array = []
    for i in range(len(mat1)):
        for j in range(len(mat1)):
            z = mat1[i][j] + mat2[i][j]
            array.append(z)
        return array