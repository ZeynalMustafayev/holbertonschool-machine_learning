#!/usr/bin/env python3
"""The shape should be returned as a list of integers"""


def add_arrays(arr1, arr2):
    """Function that adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    array = []
    for i in range(len(arr1)):
        z = arr1[i] + arr2[i]
        array.append(z)
    return array
