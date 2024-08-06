#!/usr/bin/env python3
"""Bars"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Bars"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # Fruit types and their colors
    fruit_names = ['Apples', 'Bananas', 'Oranges', 'Peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    # Number of people
    num_people = fruit.shape[1]

    # Position of the bars on the x-axis
    bar_width = 0.5
    bar_positions = np.arange(num_people)

    # Create the plot
    ax = plt.gca()

    # Stack the bars for each fruit
    bottoms = np.zeros(num_people)
    for i in range(fruit.shape[0]):
        # Use a dictionary for arguments
        bar_args = {
                    'x': bar_positions,
                    'height': fruit[i],
                    'width': bar_width,
                    'bottom': bottoms,
                    'color': colors[i],
                    'label': fruit_names[i]
                    }
        ax.bar(**bar_args)
        bottoms += fruit[i]

    # Set the labels and title
    ax.set_xlabel('Person')
    ax.set_ylabel('Quantity of Fruit')
    ax.set_title('Number of Fruit per Person')

    # Set the x-axis ticks and labels
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(['Farrah', 'Fred', 'Felicia'])

    # Set the y-axis limits and ticks
    ax.set_ylim(0, 80)
    ax.set_yticks(np.arange(0, 81, 10))

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
