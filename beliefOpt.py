"""Function used for the belief Optimization"""

import numpy as np
from utils import sigmoid


def _get_prob_matrix(alpha, q, n):
    """Get the Probability matrices"""

    # TODO investigate why some Xi values are negative
    # Might be due to poor definition of the Weight matrix
    R = np.zeros((n, n))
    Xi = np.zeros((n, n))

    beta = 1 / alpha

    # get Q
    for i in range(n):
        for j in range(n):
            R[i, j] = beta[i, j] + q[i] + q[j]
            sign = -1 if beta[i, j] < 0 else 1
            Xi[i, j] = 1 / 2 * (R[i, j] -
                                sign * np.sqrt(R[i, j] ** 2 - 4 *
                                               (1 + beta[i, j]) * q[i] * q[j]))

    return Xi


def _update_q(q, b, Xi, n, neighbors_list):
    """Update the for each node i : q_i = p_i(label = 1)"""

    for i in range(n):
        current_neighbors = neighbors_list[i]
        neighbor_count = len(current_neighbors)

        numerator = q[i] ** neighbor_count
        denominator = (1 - q[i]) ** neighbor_count
        for j in current_neighbors:
            numerator = numerator * (Xi[i, j] + 1 - q[i] - q[j])
            denominator = denominator * (q[i] - Xi[i, j])
            print("Numerator")
            print(Xi[i, j] + 1 - q[i] - q[j])
            print(Xi[i, j])
            print(q[i])
            print(q[j])
            print()

        q[i] = sigmoid(b[i] + np.log(numerator / denominator))

    return q


def belief_optimization(W, b, y, n_iter, neighbors):
    """Apply the Belief Optimization algorithm"""

    n = len(y)  # n is the number of nodes, not the grid_size

    alpha = np.exp(W) - 0.99  # avoid cases with alpha == 0
    q = sigmoid(y)

    for _ in range(n_iter):
        Xi = _get_prob_matrix(alpha, q, n)
        q = _update_q(q, b, Xi, n, neighbors)

    return q
