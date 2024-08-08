#!/usr/bin/env python3
"""poly_integral"""


def poly_integral(poly, C=0):
    """poly_integral"""
    if poly == [] or type(poly) is not list:
        return None
    if len(poly) == 1:
        return [float(C)]
    return [float(C)] + [float(poly[i] / (i + 1)) for i in range(len(poly))]
poly = [5, 3, 0, 1]
print(poly_integral(poly))