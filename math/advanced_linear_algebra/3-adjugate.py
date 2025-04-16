#!/usr/bin/env python3
"""adjugate.py"""
determinant = __import__('0-determinant').determinant
minor = __import__('1-minor').minor
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    Calculates the adjugate of a matrix.
    """
    # Validate matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 0x0 matrix
    if matrix == [[]]:
        return [[]]

    # Validate square matrix
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Base case
    if n == 1:
        return [[1]]

    # Recursive case
    cofactors = cofactor(matrix)
    adjugate_matrix = [[cofactors[j][i] for j in range(n)] for i in range(n)]
    return adjugate_matrix
