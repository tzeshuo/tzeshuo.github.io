"""Part 2: fit a line and a parabola to the given data points."""

from pathlib import Path

import matplotlib.pyplot as plt


# Given data points
data = [(0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5)]


# ============================================================
# Mean squared error
# ============================================================

def mse(parameters, model):
    """Calculate MSE for a parameterized model."""
    errors = []
    for x, y in data:
        errors.append(model(x, parameters) - y)
    return sum(error ** 2 for error in errors) / len(data)


def line(x, parameters):
    """Line model y = mx + b."""
    m, b = parameters
    return m * x + b


def parabola(x, parameters):
    """Parabola model y = ax^2 + bx + c."""
    a, b, c = parameters
    return a * x ** 2 + b * x + c


# ============================================================
# Analytical least-squares solutions
# ============================================================

def solve_linear_system(matrix, vector):
    """Solve a small linear system using Gaussian elimination."""
    matrix = [row[:] + [value] for row, value in zip(matrix, vector)]
    size = len(vector)

    for pivot in range(size):
        pivot_row = max(
            range(pivot, size),
            key=lambda row: abs(matrix[row][pivot])
        )
        matrix[pivot], matrix[pivot_row] = matrix[pivot_row], matrix[pivot]
        divisor = matrix[pivot][pivot]

        for column in range(pivot, size + 1):
            matrix[pivot][column] /= divisor

        for row in range(size):
            if row == pivot:
                continue
            multiplier = matrix[row][pivot]
            for column in range(pivot, size + 1):
                matrix[row][column] -= multiplier * matrix[pivot][column]

    return [matrix[row][size] for row in range(size)]


def analytical_fit(features):
    """Use the normal equations: (X^T X) parameters = X^T y."""
    parameter_count = len(features(0))
    x_transpose_x = [[0.0] * parameter_count for _ in range(parameter_count)]
    x_transpose_y = [0.0] * parameter_count

    for x, y in data:
        row = features(x)
        for i in range(parameter_count):
            x_transpose_y[i] += row[i] * y
            for j in range(parameter_count):
                x_transpose_x[i][j] += row[i] * row[j]

    return solve_linear_system(x_transpose_x, x_transpose_y)


# ============================================================
# Multivariable Newton-Raphson
# ============================================================

def gradient_and_hessian(parameters, features):
    """Calculate the gradient and Hessian of the MSE."""
    parameter_count = len(parameters)
    gradient = [0.0] * parameter_count
    hessian = [[0.0] * parameter_count for _ in range(parameter_count)]

    for x, y in data:
        row = features(x)
        prediction = sum(parameter * value for parameter, value in zip(parameters, row))
        error = prediction - y

        for i in range(parameter_count):
            gradient[i] += 2 * error * row[i] / len(data)
            for j in range(parameter_count):
                hessian[i][j] += 2 * row[i] * row[j] / len(data)

    return gradient, hessian


def newton_raphson_multivariable(
    initial_parameters, features, tolerance=1e-7, max_iter=100
):
    """Minimize MSE using the multivariable Newton-Raphson update."""
    parameters = initial_parameters[:]
    iterations = [parameters[:]]

    for _ in range(max_iter):
        gradient, hessian = gradient_and_hessian(parameters, features)
        change = solve_linear_system(hessian, gradient)
        next_parameters = [
            parameter - value
            for parameter, value in zip(parameters, change)
        ]
        iterations.append(next_parameters[:])

        if max(
            abs(next_value - value)
            for next_value, value in zip(next_parameters, parameters)
        ) < tolerance:
            parameters = next_parameters
            break

        parameters = next_parameters

    return parameters, iterations


# ============================================================
# Plot intermediate Newton-Raphson fits
# ============================================================

def plot_fit(model, parameters, iterations, title, filename):
    x_values = [index / 20 for index in range(61)]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        [x for x, y in data],
        [y for x, y in data],
        color="black",
        label="Data points",
        zorder=3,
    )

    for index, iteration in enumerate(iterations[:-1]):
        y_values = [model(x, iteration) for x in x_values]
        plt.plot(x_values, y_values, "--", alpha=0.5, label=f"Iteration {index}")

    final_y_values = [model(x, parameters) for x in x_values]
    plt.plot(x_values, final_y_values, label="Final fit", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ============================================================
# Run both models
# ============================================================

def main():
    media_folder = Path(__file__).parent / "media"
    line_features = lambda x: [x, 1]
    parabola_features = lambda x: [x ** 2, x, 1]

    line_analytical = analytical_fit(line_features)
    parabola_analytical = analytical_fit(parabola_features)
    line_newton, line_iterations = newton_raphson_multivariable(
        [0.0, 0.0], line_features
    )
    parabola_newton, parabola_iterations = newton_raphson_multivariable(
        [0.0, 0.0, 0.0], parabola_features
    )

    print("Line: y = mx + b")
    print("Analytical parameters:", line_analytical)
    print("Newton-Raphson parameters:", line_newton)
    print("Analytical MSE:", mse(line_analytical, line))
    print("Newton-Raphson MSE:", mse(line_newton, line))
    print()

    print("Parabola: y = ax^2 + bx + c")
    print("Analytical parameters:", parabola_analytical)
    print("Newton-Raphson parameters:", parabola_newton)
    print("Analytical MSE:", mse(parabola_analytical, parabola))
    print("Newton-Raphson MSE:", mse(parabola_newton, parabola))

    plot_fit(
        line,
        line_newton,
        line_iterations,
        "Line fit: y = mx + b",
        media_folder / "part2-line-fit.png",
    )
    plot_fit(
        parabola,
        parabola_newton,
        parabola_iterations,
        "Parabola fit: y = ax^2 + bx + c",
        media_folder / "part2-parabola-fit.png",
    )


if __name__ == "__main__":
    main()
