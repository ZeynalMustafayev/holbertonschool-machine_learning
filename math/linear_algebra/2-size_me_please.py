#!/usr/bin/env python3
def matrix_shape(matrix):
    size = []
    while isinstance(matrix, list):
        size.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return size
