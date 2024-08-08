#!/usr/bin/env python3
"""poly_integral"""


def poly_integral(poly, C=0):
    """poly_integral"""
    if poly == [] or type(poly) is not list:
        return None
    if len(poly) == 1:
        return [C]
    integral = []
    integral.append(C)
    for i in range(len(poly)):
        x = poly[i] / (i + 1)
        integral.append(int(x) if x.is_integer() else x)
    return integral
