#!/usr/bin/env python3
"""Module that concatenates two arrays"""


def cat_arrays(arr1, arr2):
    new_list = []
    sum = arr1 + arr2
    for i in sum:
        new_list.append(i)
    return new_list
