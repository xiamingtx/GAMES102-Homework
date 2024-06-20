import numpy as np
import matplotlib.pyplot as plt


# Calculate the angle between two vectors
def compute_angle(v1, v2):
    return np.arccos(np.clip(np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))


def uniform_parameterization(points):
    n = len(points)
    return np.linspace(0, 1, n)


def chordal_parameterization(points):
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    return cumulative_distances / cumulative_distances[-1]


def centripetal_parameterization(points):
    distances = np.sqrt(np.linalg.norm(np.diff(points, axis=0), axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    return cumulative_distances / cumulative_distances[-1]


def foley_parameterization(points):
    n = len(points)
    # Simplified processing, return uniform parameterization when there are fewer than 3 points
    if n < 3:
        return np.linspace(0, 1, n)

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    angles = []
    for i in range(1, n - 1):
        v1 = points[i - 1] - points[i]
        v2 = points[i + 1] - points[i]
        angle = compute_angle(v1, v2)
        angles.append(np.pi - angle if angle > np.pi / 2 else np.pi / 2)

    foley_distances = [distances[0] * (1 + 1.5 * angles[0] * distances[1] / (distances[0] + distances[1]))]
    for i in range(1, n - 2):
        d = distances[i] * (1 + 1.5 * angles[i - 1] * distances[i - 1] / (distances[i - 1] + distances[i]) +
                            1.5 * angles[i] * distances[i + 1] / (distances[i] + distances[i + 1]))
        foley_distances.append(d)
    foley_distances.append(distances[-2] * (1 + 1.5 * angles[-1] * distances[-2] / (distances[-3] + distances[-2])))

    cumulative_distances = np.insert(np.cumsum(foley_distances), 0, 0)
    return cumulative_distances / cumulative_distances[-1]


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


if __name__ == "__main__":
    points = np.array([[0, 0], [1, 2], [3, 1], [4, 5], [6, 4]])

    # polynomial fitting
    poly_coefficients = polynomial_fitting(points)
    # Generate points on the fitted curve
    x_fit = np.linspace(min(points[:, 0]), max(points[:, 0]), 100)
    y_fit = np.polyval(poly_coefficients[::-1], x_fit)

    # Plotting the fitted curves and raw points
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 2, 1)
    plt.plot(x_fit, y_fit, label='Polynomial Fit')
    plt.scatter(points[:, 0], points[:, 1], color='red', label='Original Points')
    plt.title('Polynomial Fitting of Points')
    plt.legend()

    # Using different parameterization methods to generate the parameter t
    uniform_t = uniform_parameterization(points)
    chordal_t = chordal_parameterization(points)
    centripetal_t = centripetal_parameterization(points)
    foley_t = foley_parameterization(points)

    # Plotting t-value curves for different parameterization methods
    plt.subplot(2, 2, 2)
    plt.plot(points[:, 0], uniform_t, label='Uniform')
    plt.plot(points[:, 0], chordal_t, label='Chordal')
    plt.plot(points[:, 0], centripetal_t, label='Centripetal')
    plt.plot(points[:, 0], foley_t, label='Foley')
    plt.title('Parameterization Methods Comparison')
    plt.xlabel('x-coordinate')
    plt.ylabel('Parameter t')
    plt.legend()

    # Calculate and plot point spacing for each parametric method
    plt.subplot(2, 2, 3)
    for t, label in zip([uniform_t, chordal_t, centripetal_t, foley_t], ['Uniform', 'Chordal', 'Centripetal', 'Foley']):
        distances = np.diff(np.interp(t, np.linspace(0, 1, 100), y_fit))
        plt.plot(t[:-1], distances, label=f'{label} Distances')
    plt.title('Distance between Points in Parameterization')
    plt.xlabel('Parameter t')
    plt.ylabel('Distance')
    plt.legend()

    plt.tight_layout()
    plt.show()
