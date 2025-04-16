#!/usr/bin/env python3
"""
determinant.py

"""


def determinant(matrix):
    """
    Calculates the determinant of a square matrix.
    """
    # Validate matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 0x0 matrix
    if matrix == [[]]:
        return 1

    # Validate square matrix
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base cases
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    # Recursive case
    det = 0
    for col in range(n):
        # Build the minor matrix
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]
        cofactor = ((-1) ** col) * matrix[0][col] * determinant(minor)
        det += cofactor
    return det
