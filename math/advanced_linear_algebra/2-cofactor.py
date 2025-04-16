#!/usr/bin/env python3
"""minor.py"""


determinant = __import__('0-determinant').determinant


def minor(matrix):
    """
    Calculates the minor of a square matrix.
    """
    # Validate matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 0x0 matrix
    if matrix == [[]]:
        return [[]]

    # Validate square matrix
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case
    if n == 1:
        return [[1]]

    # Recursive case
    minors = []
    for i in range(n):
        minors.append([])
        for j in range(n):
            minor = [r[:j] + r[j+1:] for k, r in enumerate(matrix) if k != i]
            minors[i].append(determinant(minor))
    return minors
