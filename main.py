"""main file for the project"""

import numpy as np
from utils import generate_grid, draw_energy_comparison_plot
from beliefopt import belief_optimization
from junctiontree import IsingJunction


# TODO implement the junction tree algorithm for comparison of results.

def grid_example(grid_size=10, weight_scale: float = 1., bias_scale: float = .1, n_iter=2):
    """Runs the basic grid example"""
    if grid_size < 2:
        raise ValueError("Grid Size should be at least 2")
    if weight_scale <= 0 or bias_scale <= 0:
        raise ValueError("Scales should be higher than 0.0")

    neighbors, Weight = generate_grid(grid_size)
    Weight = Weight * weight_scale
    bias = np.random.randn(grid_size * grid_size) * bias_scale
    q = np.random.random((grid_size * grid_size))

    junction_alg = IsingJunction(q, Weight, grid_size)

    for i in range(grid_size * grid_size):
        neighbors_weight = sum([Weight[i, neighbor] for neighbor in neighbors[i]])
        bias[i] -= 1 / 2 * neighbors_weight

    result_fixed, free_energy_fixed = belief_optimization(Weight, bias, q, n_iter, neighbors, use_grad=False)
    result_grad, free_energy_grad = belief_optimization(Weight, bias, q, n_iter, neighbors, use_grad=True)
    result_new_grad, free_energy_new_grad = belief_optimization(Weight, bias, q, n_iter, neighbors, use_new_grad=True)

    print(result_fixed)
    print(result_grad)
    print(result_new_grad)


    draw_energy_comparison_plot(free_energy_grad, free_energy_new_grad, "Gradient", "Our Version", "grad_vs_ours.pdf")
    draw_energy_comparison_plot(free_energy_grad, free_energy_fixed, "Gradient", "Fixed Point", "grad_vs_fixed.pdf")


if __name__ == "__main__":
    grid_example(grid_size=8, n_iter=500)
