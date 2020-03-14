"""Function used for the belief Optimization"""

import numpy as np
from utils import sigmoid


def _get_prob_matrix(alpha, q, n):
    """Get the Probability matrices"""
    Q = np.zeros((n, n))
    Xi = np.zeros((n, n))

    # get Q
    for i in range(n):
        for j in range(n):
            # TODO Check order of Q and Xi updates
            Q[i, j] = 1 + alpha[i, j] * q[i] + alpha[i, j] * q[j]
            Xi[i, j] = 1 / alpha[i, j] * (
                    Q[i, j] - np.sqrt(Q[i, j] ** 2 - 4 * alpha[i, j] *
                                      (1 + alpha[i, j]) * q[i] * q[j]))

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
        q[i] = sigmoid(b[i] + np.log(numerator / denominator))

    return q


def belief_optimization(W, b, y, n_iter, neighbors):
    """Apply the Belief Optimization algorithm"""

    n = len(y)

    alpha = np.exp(W) - 0.99 # avoid cases with alpha == 0
    q = sigmoid(y)

    for _ in range(n_iter):
        Xi = _get_prob_matrix(alpha, q, n)
        print(Xi)

        q = _update_q(q, b, Xi, n, neighbors)

    return q
