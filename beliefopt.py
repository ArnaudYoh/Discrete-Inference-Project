"""Functions used for the belief Optimization"""

import numpy as np
from utils import sigmoid


def _get_prob_matrix(alpha, q, n):
    """Get the Probability matrices"""
    R = np.zeros((n, n))
    Xi = np.zeros((n, n))

    # get Q
    for i in range(n):
        for j in range(n):
            if alpha[i, j] == 0:
                continue
            beta = 1 / alpha[i, j]
            R[i, j] = beta + q[i] + q[j]
            sign = -1 if beta < 0 else 1
            Xi[i, j] = 1 / 2 * (R[i, j] -
                                sign * np.sqrt(R[i, j] ** 2 - 4 *
                                               (1 + beta) * q[i] * q[j]))
            if Xi[i, j] < q[i] + q[j] - 1:
                raise ValueError("This is behaviour is unexpected, please contact your nearest coding monkey")

    return Xi


def _update_q(q, b, Xi, n, neighbors_list):
    """Update the for each node i : q_i = p_i(label = 1)"""

    updated_q = np.copy(q)

    for i in range(n):
        current_neighbors = neighbors_list[i]
        neighbor_count = len(current_neighbors)

        numerator = q[i] ** neighbor_count
        denominator = (1 - q[i]) ** neighbor_count
        for j in current_neighbors:
            numerator = numerator * (Xi[i, j] + 1 - q[i] - q[j])
            denominator = denominator * (q[i] - Xi[i, j])

        updated_q[i] = sigmoid(b[i] + np.log(numerator / denominator))

    return updated_q


def belief_optimization(W, b, q, n_iter, neighbors):
    """Apply the Belief Optimization algorithm"""

    n = len(q)  # n is the number of nodes, not the grid_size
    alpha = np.exp(W) - 1.0

    for _ in range(n_iter):
        Xi = _get_prob_matrix(alpha, q, n)
        print("Xi", np.reshape(Xi, (n, n)))
        q = _update_q(q, b, Xi, n, neighbors)

    return q
