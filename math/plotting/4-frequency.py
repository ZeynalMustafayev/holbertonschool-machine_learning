#!/usr/bin/env python3
"""Frequency"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Frequency"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor="black")
    plt.show()
