#!/usr/bin/env python3
"""sum total numbers from 1 to n squared"""


def summation_i_squared(n):
    """sum total numbers from 1 to n squared"""
    if n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
