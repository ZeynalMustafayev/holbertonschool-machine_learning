#!/usr/bin/env python3
"""poly_derivative"""


def poly_derivative(poly):
    """poly_derivative"""
    if poly == [] or type(poly) is not list:
        return None
    if len(poly) == 1:
        return [0]
    return [poly[i] * i for i in range(1, len(poly))]
