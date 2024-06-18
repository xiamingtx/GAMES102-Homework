#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:fit.py
# author:xm
# datetime:2024/6/18 21:20
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
sigma = 1
lam = 0.01


def solve(A, b):
    """
    Use SVD to solve equation: Ax = b
    """
    if A.shape[0] >= A.shape[1]:
        # overdetermined equation. In this case, there is usually no exact solution, so the goal is to find
        # a least squares solution, i.e. one that minimizes the sum of squared residuals.
        u, s, vt = np.linalg.svd(A)
        # Moore-Penrose inverse: (A+) = V(Sigma+)(U.T) -> x = (A+) * b = V(Sigma+)(U.T) * b, Sigma+ = matrix [1 / sigma]
        x = np.dot(vt.T, (np.dot(u.T, b)[:s.shape[0]] / s))
    else:
        # underdetermined equation. Finding minimum norm solutions via singular value decomposition (SVD)
        start = A.shape[1] - A.shape[0]
        # Cut the A matrix into squares matrix to simplify the problem so that we can apply SVD to find the solution.
        A = A[:, start:]
        # SVD
        u, s, vt = np.linalg.svd(A)
        x = np.dot(vt.T, (np.dot(u.T, b)[:s.shape[0]] / s))
        # Match the dimension of the solution to the number of unknowns in the original problem.
        x = np.concatenate((np.zeros(start), x))

    return x


# interpolation (polynomial fitting)
def polynomial_fitting(points):
    n = len(points)
    x_array = points[:, 0]
    y_array = points[:, 1]

    # construct the Vandermonde Matrix
    A = np.zeros((n, n))
    x_power = np.ones_like(x_array)
    for i in range(n):
        # Initially, set col[0] as 1, as any number raised to the 0th power is 1.
        A[:, i] = x_power
        # During the loop, the x value of each column is gradually raised to a power.
        # The first column is x^0, the second column is x^1, and so on to x^(n-1)
        x_power = np.multiply(x_power, x_array)
    r = solve(A, y_array)
    return r


# interpolation (Gaussian fitting)
def gaussian_fitting(points):
    n = len(points)
    x_array = points[:, 0]
    y_array = points[:, 1]
    # Store Gaussian basis functions and constant terms
    A = np.zeros((n, n + 1))
    # The first column is set to 1, which corresponds to the coefficient of the constant term
    A[:, 0] = np.ones_like(x_array)

    # Construct a Gaussian function for each data point
    for i in range(n):
        # Calculate the difference between the current point and all other points
        tmp_x = x_array - x_array[i]
        # Calculate the Gaussian function value centered at the current point and store it in matrix A
        A[:, i + 1] = np.exp(- np.power(tmp_x, 2) / (2 * sigma * sigma))
    r = solve(A, y_array)
    return r


# approximate fitting (Least Square Method)
def regression(points, n=3):
    point_size = len(points)
    x_array = points[:, 0]
    y_array = points[:, 1]
    # Maximum power is fixed
    A = np.zeros((point_size, n))
    x_power = np.ones_like(x_array)
    for i in range(n):
        A[:, i] = x_power
        x_power = np.multiply(x_power, x_array)
    r = solve(A, y_array)

    return r


# approximate fitting (Ridge Regression)
def ridge_regression(points, n=3, lam=0.1):
    point_size = len(points)
    x_array = points[:, 0]
    y_array = points[:, 1]
    A = np.zeros((point_size, n))
    x_power = np.ones_like(x_array)
    for i in range(n):
        A[:, i] = x_power
        x_power = np.multiply(x_power, x_array)

    # Construct Ridge Matrix (A^T A + λI)
    I = np.eye(n)
    ridge_matrix = np.dot(A.T, A) + lam * I
    # A^T * Y
    A_T_y = np.dot(A.T, y_array)
    # solve Ax = b
    r = solve(ridge_matrix, A_T_y)

    return r


if __name__ == "__main__":
    # generate points
    points = np.array([[0.50, 0.40],
                       [0.80, 0.30],
                       [0.30, 0.80],
                       [-0.40, 0.30],
                       [-0.30, 0.70]])

    plt.xlim((-5, 5))
    plt.ylim((-2, 2))
    # draw original points
    points = np.array(points)
    # print(points)
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y, color='red')

    x_axis = np.linspace(-5, 5, 100)

    # draw polynomial fitting
    w_1 = polynomial_fitting(points)

    x_matrix = np.zeros((x_axis.shape[0], w_1.shape[0]))
    x_power = np.ones_like(x_axis)

    for i in range(w_1.shape[0]):
        x_matrix[:, i] += x_power
        x_power = np.multiply(x_power, x_axis)
    y1 = np.matmul(x_matrix, w_1)
    plt.plot(x_axis, y1, label="polynomial interpolation")

    # draw gaussian fitting
    w_2 = gaussian_fitting(points)

    x_matrix = np.zeros((x_axis.shape[0], w_2.shape[0]))
    x_matrix[:, 0] = np.ones_like(x_axis)
    for i in range(0, w_2.shape[0] - 1):
        tmp_x = x_axis - np.ones_like(x_axis) * x[i]
        x_matrix[:, i + 1] = np.exp(- np.power(tmp_x, 2) / (2 * sigma * sigma))

    y2 = np.matmul(x_matrix, w_2)
    plt.plot(x_axis, y2, label="gaussian interpolation")

    # draw regression fitting
    # number of parameters
    n = 5

    w_3 = regression(points, n)

    x_matrix = np.zeros((x_axis.shape[0], w_3.shape[0]))
    x_power = np.ones_like(x_axis)

    for i in range(w_3.shape[0]):
        x_matrix[:, i] += x_power
        x_power = np.multiply(x_power, x_axis)
    y3 = np.matmul(x_matrix, w_3)
    plt.plot(x_axis, y3, label="regression")

    # draw ridge regression

    w_4 = ridge_regression(points, n, lam)

    x_matrix = np.zeros((x_axis.shape[0], w_4.shape[0]))
    x_power = np.ones_like(x_axis)

    for i in range(w_4.shape[0]):
        x_matrix[:, i] += x_power
        x_power = np.multiply(x_power, x_axis)
    y4 = np.matmul(x_matrix, w_4)
    plt.plot(x_axis, y4, label="ridge regression")

    print(np.linalg.norm(w_3), np.linalg.norm(w_4))
    plt.legend()
    plt.show()
