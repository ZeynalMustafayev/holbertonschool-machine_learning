#!/usr/bin/env python3
"""Bars function"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Bars function"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ["Farrah", "Fred", "Felicia"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]

    # Plotting the stacked bar graph
    plt.bar(people, fruit[0], color=colors[0], width=0.5, label="apples")
    plt.bar(people, fruit[1], color=colors[1],
            width=0.5, bottom=fruit[0], label="bananas")
    plt.bar(people, fruit[2], color=colors[2],
            width=0.5, bottom=fruit[0] + fruit[1], label="oranges")
    plt.bar(people, fruit[3], color=colors[3],
            width=0.5, bottom=fruit[0] + fruit[1] + fruit[2], label="peaches")

    # Adding labels, title, and legend
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()
    plt.show()
