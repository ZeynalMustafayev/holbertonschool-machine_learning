#!/usr/bin/env python3
"""sum total numbers from 1 to n squared"""


def summation_i_squared(n):
    """sum total numbers from 1 to n squared"""
    if n < 0:
        return None
    sum = 0
    for i in range(n + 1):
        sum += i**2
    return sum
