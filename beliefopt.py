"""Functions used for the belief Optimization"""

import numpy as np
from utils import sigmoid


def _get_Xi(alpha, q, n):
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
                raise ValueError("This is behaviour is unexpected, we cannot have Xi[i, j] < q[i] + q[j] - 1")

            if Xi[i, j] > q[i]:
                raise ValueError("This is behaviour is unexpected, we cannot have Xi[i, j] > q[i]")

            if Xi[i, j] > q[j]:
                raise ValueError("This is behaviour is unexpected, we cannot have Xi[i, j] > q[j]")

    return Xi


def _update_q_gradient2(W, q, b, Xi, n, neighbors_list, alpha):
    """Update the for each node i : q_i = p_i(label = 1)"""
    updated_q = np.copy(q)

    for i in range(n):
        current_neighbors = neighbors_list[i]
        neighbor_count = len(current_neighbors)
        if neighbor_count == 0:
            continue

        numerator = (1 - updated_q[i]) ** (neighbor_count - 1)
        denominator = updated_q[i] ** (neighbor_count - 1)
        for j in current_neighbors:
            numerator = numerator * (updated_q[i] - Xi[i, j])
            denominator = denominator * (Xi[i, j] + 1 - updated_q[i] - updated_q[j])

        gradient = (-b[i] + np.log(numerator / denominator)) * updated_q[i] * (1 - updated_q[i])
        updated_q[i] -= gradient
        Xi = _get_Xi(alpha, updated_q, n)
        # print(compute_bethe_free_energy(W, Xi, updated_q, b, neighbors_list, n))

    return updated_q


def _update_q_gradient(q, b, Xi, n, neighbors_list):
    """Update the for each node i : q_i = p_i(label = 1)"""
    updated_q = np.copy(q)

    for i in range(n):
        current_neighbors = neighbors_list[i]
        neighbor_count = len(current_neighbors)
        if neighbor_count == 0:
            continue

        numerator = (1 - q[i]) ** (neighbor_count - 1)
        denominator = q[i] ** (neighbor_count - 1)
        for j in current_neighbors:
            numerator = numerator * (q[i] - Xi[i, j])
            denominator = denominator * (Xi[i, j] + 1 - q[i] - q[j])

        gradient = (-b[i] + np.log(numerator / denominator)) * q[i] * (1 - q[i])
        updated_q[i] -= gradient

    return updated_q


def _update_q_fixed_point(q, b, Xi, n, neighbors_list):
    """Update the for each node i : q_i = p_i(label = 1)"""

    updated_q = np.copy(q)

    for i in range(n):
        current_neighbors = neighbors_list[i]
        neighbor_count = len(current_neighbors)
        if neighbor_count == 0:
            continue

        numerator = q[i] ** neighbor_count
        denominator = (1 - q[i]) ** neighbor_count
        for j in current_neighbors:
            numerator = numerator * (Xi[i, j] + 1 - q[i] - q[j])
            denominator = denominator * (q[i] - Xi[i, j])

        updated_q[i] = sigmoid(b[i] + np.log(numerator / denominator))

    return updated_q


def compute_bethe_free_energy(W, Xi, q, b, neighbors_list, n):
    E_nodes = 0
    E_edges = 0
    S1 = 0
    S2 = 0
    for i in range(n):
        E_nodes -= b[i] * q[i]
        current_neighbors = neighbors_list[i]
        neighbor_count = len(current_neighbors)
        S1 -= (1 - neighbor_count) * (q[i] * np.log(q[i]) + (1 - q[i]) * np.log(1 - q[i]))
        for j in current_neighbors:
            E_edges -= W[i, j] * Xi[i, j]
            S2 -= Xi[i, j] * np.log(Xi[i, j]) + (Xi[i, j] + 1 - q[i] - q[j]) * np.log((Xi[i, j] + 1 - q[i] - q[j])) + \
                (q[i] - Xi[i, j]) * np.log(q[i] - Xi[i, j]) + (q[j] - Xi[i, j]) * np.log(q[j] - Xi[i, j])

    # We counted each edge twice
    E_edges = E_edges / 2
    S2 = S2 / 2

    return E_nodes + E_edges + S1 + S2


def belief_optimization(W, b, q, n_iter, neighbors, use_grad=False):
    """Apply the Belief Optimization algorithm"""

    n = len(q)  # n is the number of nodes, not the grid_size
    alpha = np.exp(W) - 1.0

    for i in range(n_iter):
        Xi = _get_Xi(alpha, q, n)
        print("iteration", i)
        print("Après l'update du nouveau Xi", compute_bethe_free_energy(W, Xi, q, b, neighbors, n))
        if use_grad:
            q = _update_q_gradient(q, b, Xi, n, neighbors)
        else:
            q = _update_q_fixed_point(q, b, Xi, n, neighbors)
        print("Avant l'update du nouveau Xi", compute_bethe_free_energy(W, Xi, q, b, neighbors, n))

    return q
