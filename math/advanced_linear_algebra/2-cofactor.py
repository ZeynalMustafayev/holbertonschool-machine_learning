#!/usr/bin/env python3
"""cofactor.py"""


determinant = __import__('0-determinant').determinant
minor = __import__('1-minor').minor

def cofactor(matrix):
    """
    Calculates the cofactor of a matrix.
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
        raise ValueError("matrix must be a non-empty square matrix")

    # Base case
    if n == 1:
        return [[1]]

    # Recursive case
    cofactors = []
    for i in range(n):
        cofactors.append([])
        for j in range(n):
            minor_matrix = [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]
            cofactors[i].append(((-1) ** (i + j)) * determinant(minor_matrix))
    return cofactors
