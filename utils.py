"""File for utility function and classes"""

import numpy as np


def sigmoid(x):
    """Computes the sigmoid function over a numpy array"""
    return 1 / (1 + np.exp(-x))


def generate_grid(n, p0=0.5, p1=0.5):
    """Create a nxn grid defined by its neighborhood dict and the symmetric weight matrix"""

    if p0 + p1 != 1:
        raise ValueError("p0 and p1 should add up to 1")

    # Note: the entries of the weight matrix are pairs of node rather than a single node
    neighbors = {}
    W = np.zeros((n * n, n * n))
    for i in range(n * n):
        neighbors[i] = []
        if i == 0:
            continue
        if i > n - 1:
            neighbors[i].append(i - n)
            W[i, i - n] += 1
        if i % n > 0:
            neighbors[i].append(i - 1)
            W[i, i - 1] += 1
        if i % n < n - 1:
            neighbors[i].append(i + 1)
            W[i, i + 1] += 1
        if i < n * (n - 1):
            neighbors[i].append(i + n)
            W[i, i + n] += 1

    return neighbors, W
