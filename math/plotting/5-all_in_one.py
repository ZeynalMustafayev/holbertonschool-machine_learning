#!/usr/bin/env python3
"""All in One"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def all_in_one():
    """All in One"""
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Creating the GridSpec layout
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("All in One")
    gs = GridSpec(3, 2, figure=fig)

    # Plot 1: y0 vs. index
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(y0, c="red")
    ax1.set_xlim([0, 10])
    ax1.set_xticks(np.arange(0, 11, 2))
    ax1.set_yticks(np.arange(0, 1001, 500))

    # Plot 2: Scatter plot of height vs weight
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(x1, y1, c='magenta')
    ax2.set_xlabel('Height (in)')
    ax2.set_ylabel('Weight (lbs)')
    ax2.set_title("Men's Height vs Weight")

    # Plot 3: Exponential decay of C-14
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x2, y2)
    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel("Fraction Remaining")
    ax3.set_title("Exponential Decay of C-14")
    ax3.set_yscale("log")
    ax3.set_xlim([0, 28650])

    # Plot 4: Exponential decay of Radioactive Elements
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x3, y31, linestyle="--", c="red", label="C-14")
    ax4.plot(x3, y32, linestyle="-", c="green", label="Ra-226")
    ax4.set_xlabel("Time (years)")
    ax4.set_ylabel("Fraction Remaining")
    ax4.set_title("Exponential Decay of Radioactive Elements")
    ax4.set_xlim([0, 20000])
    ax4.set_ylim([0, 1])
    ax4.legend()

    # Plot 5: Histogram of student grades
    ax5 = fig.add_subplot(gs[2, :])
    ax5.hist(student_grades, bins=range(0, 110, 10), edgecolor="black")
    ax5.set_xlabel("Grades")
    ax5.set_ylabel("Number of Students")
    ax5.set_title("Project A")
    ax5.set_xlim(0, 100)
    ax5.set_ylim(0, 30)
    ax5.set_xticks(np.arange(0, 110, 10))

    plt.tight_layout()
    plt.show()

all_in_one()
