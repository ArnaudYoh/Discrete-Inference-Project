"""main file for the project"""

import numpy as np
from utils import generate_grid
from beliefOpt import belief_optimization

# TODO implement the junction tree algorithm for comparison of results.

def grid_example(grid_size=10, scale: float = 1.):
    """Runs the basic grid example"""
    if grid_size < 2:
        raise ValueError("Grid Size should be at least 2")
    if scale > 1.0:
        raise ValueError("Scale should be at most 1.0")

    neighbors, Weight = generate_grid(grid_size)
    Weight = Weight * scale

    q = np.random.random(grid_size * grid_size)

    bias = np.random.randn(grid_size * grid_size)
    for i in range(grid_size):
        neighbors_weight = sum([Weight[i, neighbor] for neighbor in neighbors[i]])
        bias[i] -= 1 / 2 * neighbors_weight

    print(belief_optimization(Weight, bias, q, 1, neighbors))


if __name__ == "__main__":
    grid_example()
