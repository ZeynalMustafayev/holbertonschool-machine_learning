#!/usr/bin/env python3


def determinant(matrix):
    """
    Calculate the determinant of a square matrix using recursion.
    """
    if not matrix:
        raise ValueError("Matrix is empty")
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix is not square")

    n = len(matrix)

    # Base case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case for larger matrices
    det = 0
    for c in range(n):
        minor = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(minor)

    return det
