#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:subdivision_curves.py
# author:xm
# datetime:2024/6/22 13:09
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import numpy as np
import matplotlib.pyplot as plt


def chaikin(points, n_iterations):
    for _ in range(n_iterations):
        new_points = []
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_points.extend([q, r])
        points = np.vstack([new_points, points[-1]])
    return points


def cubic_bspline_subdivision(points, n_iterations):
    for _ in range(n_iterations):
        new_points = [points[0]]
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            p0_ = (1 / 8) * p0 + (6 / 8) * p1 + (1 / 8) * points[min(i + 2, len(points) - 1)]
            new_points.extend([p0_, p1])
        points = np.array(new_points)
    return points


def four_point_subdivision(points, n_iterations):
    for _ in range(n_iterations):
        new_points = [points[0]]
        for i in range(1, len(points) - 1):
            p_minus = points[i - 1]
            p = points[i]
            p_plus = points[i + 1]
            new_p = (-1 / 16) * p_minus + (9 / 16) * p + (9 / 16) * p_plus + (-1 / 16) * points[
                min(i + 2, len(points) - 1)]
            new_points.extend([p, new_p])
        new_points.append(points[-1])
        points = np.array(new_points)
    return points


if __name__ == '__main__':
    # Control Points
    points = np.array([[0, 0], [1, 2], [2, 3], [4, 1], [5, 3], [6, 0]])
    n_iterations = 3

    # fit subdivison curves
    subdiv_points_chaikin = chaikin(points, n_iterations)
    subdiv_points_bspline = cubic_bspline_subdivision(points, n_iterations)
    subdiv_points_four_point = four_point_subdivision(points, n_iterations)

    # Plot Chaikin Method
    plt.figure(figsize=(10, 6))
    plt.plot(points[:, 0], points[:, 1], 'bo-', label='Original Points')
    plt.plot(subdiv_points_chaikin[:, 0], subdiv_points_chaikin[:, 1], 'ro-', label='Chaikin Subdivision')
    plt.title('Chaikin Method')
    plt.legend()
    plt.show()

    # Plot Cubic B-spline Method
    plt.figure(figsize=(10, 6))
    plt.plot(points[:, 0], points[:, 1], 'bo-', label='Original Points')
    plt.plot(subdiv_points_bspline[:, 0], subdiv_points_bspline[:, 1], 'go-', label='Cubic B-spline Subdivision')
    plt.title('Cubic B-spline Method')
    plt.legend()
    plt.show()

    # Plot 4-Point Method
    plt.figure(figsize=(10, 6))
    plt.plot(points[:, 0], points[:, 1], 'bo-', label='Original Points')
    plt.plot(subdiv_points_four_point[:, 0], subdiv_points_four_point[:, 1], 'mo-', label='4-Point Subdivision')
    plt.title('4-Point Method')
    plt.legend()
    plt.show()
