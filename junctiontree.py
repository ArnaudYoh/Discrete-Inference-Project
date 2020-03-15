"""File for applying the junction tree algorithm"""

import numpy as np


class IsingJunction:

    def __init__(self, E, W, n):
        self.n = n  # length of the grid
        self.omega = 2 ** n  # nb of configurations by column of the grid
        self.E = E  # vector of energies
        self.W = W  # matrix of correlations energies
        self.forward_messages = np.ones((self.n, self.omega))  # to stock forward matrix

    def int2config(self, s):
        """create the configuration of state s for a column"""
        config = np.zeros(self.n)
        interm = list(bin(s)[2:])
        for j in range(len(interm)):
            config[j] = int(interm[- j - 1])
        return config

    def e_junction(self, m, s):
        """compute the energy of the column m in the state s"""
        config = self.int2config(s)
        e = 0
        for i in range(self.n):
            e += config[i] * self.E[self.n * m + i]
        for i in range(self.n - 1):
            indicator = 1 if config[i] == config[i + 1] else 0
            e += indicator * self.W[self.n * m + i, self.n * m + i + 1]
        return e

    def w_junction(self, m, s1, s2):
        """compute the correlation energies for column m in state s1 and column m+1 in state s2"""
        config1 = self.int2config(s1)
        config2 = self.int2config(s2)
        w = 0
        for i in range(self.n):
            indicator = 1 if config1[i] == config2[i] else 0
            w += indicator * self.W[self.n * m + i, self.n * m + i + 1]
        return w

    def update_forward_messages(self):
        """use formula to compute all forward messages"""
        for m in range(self.n - 1):
            for s2 in range(self.omega):
                for s1 in range(self.omega):
                    energy = self.e_junction(m, s1) * self.w_junction(m, s1, s2) * self.forward_messages[m][s1]
                    self.forward_messages[m + 1][s2] += energy

    def compute_Z(self):
        """compute the partition function. We use the forward messages at the last columns so we do
        not have to compute the backward messages"""
        self.update_forward_messages()
        Z = 0
        for s in range(self.omega):
            Z += self.e_junction(self.n - 1, s) * self.forward_messages[self.n - 1][s]
        return Z, self.E
