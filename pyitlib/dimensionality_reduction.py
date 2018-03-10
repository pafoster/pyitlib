# -*- coding: utf-8 -*-
# The MIT License (MIT)
#
# Copyright (c) 2016 Peter Foster <pyitlib@gmx.us>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements dimensionality reduction methods, chiefly for discrete
random variables.

"""

# TODO Helper function for processing input variables. It should sort, then
# encode variables with an increasing sequence of integers. The integers should
# then be converted to base 2 (using boolean arrays).

# TODO Start with MRMR algorithm. The output variable cannot be continuous, but
# input variables may be continuous. Feature selection operates on the binary
# expansion, selected indices should be mapped back to input dimensions
# (n-to-1). MRMR Should also allow input features to be transformed. (Use
# similar interface to sklearn?)

import numpy as np
from discrete_random_variable import information_mutual
from discrete_random_variable import entropy
import pdb

# TODO Use consistent array format - first dimension is variable, second is
# observation

# Dummy data
N = 1E1
Y = np.random.normal(size=(N,)) > 0
X = np.random.normal(size=(N, 4)) > 0
X[:, 0] = Y

# Cluster assignments
D = np.ones((N,))


def cost_function(X, Z, Y, lamb):
    # Stochastically encode X and Y based on Z
    X_prime = np.empty_like(X)
    Y_prime = np.empty_like(Y)
    for i in np.arange(2):
        I = np.where(Z == i)[0]
        if I.size > 0:
            X_prime[I, :] = X[np.roll(I, 1), :]
            Y_prime[I] = Y[np.roll(I, 1)]

    MI = - lamb * information_mutual(np.ravel(Y), np.ravel(Y_prime))

    return MI


def coordinate_descent(X, Z, Y, lamb=100):
    cost = np.inf
    for i in np.arange(300):
        # Random shuffle -- TODO why does this break things?
        I = np.random.permutation(X.shape[0])
        X = X[I, :]
        Z = Z[I]
        Y = Y[I]

        for j in np.arange(X.shape[0]):
            Z[j] = False
            MI = cost_function(X, Z, Y, lamb)
            Z[j] = True
            MI_2 = cost_function(X, Z, Y, lamb)
            if MI < MI_2:
                Z[j] = False
                cost = MI
            else:
                Z[j] = True
                cost = MI_2
        print(str(i) + str(cost))
    return cost


coordinate_descent(X, D, Y)
