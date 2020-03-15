"""main file for the project"""

import numpy as np
from utils import generate_grid
from beliefopt import belief_optimization


# TODO implement the junction tree algorithm for comparison of results.

def grid_example(grid_size=10, weight_scale: float = 1., bias_scale: float = .1):
    """Runs the basic grid example"""
    if grid_size < 2:
        raise ValueError("Grid Size should be at least 2")
    if weight_scale <= 0 or bias_scale <= 0 :
        raise ValueError("Scales should be higher than 0.0")

    neighbors, Weight = generate_grid(grid_size)
    Weight = Weight * weight_scale
    bias = np.zeros(grid_size * grid_size)
    #bias = np.random.randn(grid_size * grid_size) * bias_scale

    q = np.random.choice([0.1], (grid_size * grid_size))

    for i in range(grid_size * grid_size):
        neighbors_weight = sum([Weight[i, neighbor] for neighbor in neighbors[i]])
        bias[i] -= 1 / 2 * neighbors_weight

    result = belief_optimization(Weight, bias, q, 1, neighbors)

    print("Bias", np.reshape(bias, (grid_size, grid_size)))
    print("OG q", np.reshape(q, (grid_size, grid_size)))
    print("Updated q", np.reshape(result, (grid_size, grid_size)))


if __name__ == "__main__":
    grid_example()
