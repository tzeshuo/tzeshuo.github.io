"""Part 2: numerical line and parabola fitting."""

from pathlib import Path

import matplotlib.pyplot as plt


# Given data points
DATA = [(0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5)]


# ============================================================
# Models and mean squared error
# ============================================================

def line(x, parameters):
    """Line model: y = m*x + b."""
    m, b = parameters
    return m * x + b


def parabola(x, parameters):
    """Parabola model: y = a*x^2 + b*x + c."""
    a, b, c = parameters
    return a * x**2 + b * x + c


def mean_squared_error(parameters, model):
    """Calculate the MSE for a model and its parameters."""
    total_error = 0.0

    for x, y in DATA:
        error = model(x, parameters) - y
        total_error += error**2

    return total_error / len(DATA)


# ============================================================
# Multivariable Newton-Raphson method
# ============================================================


def features_for_line(x):
    return [x, 1]


def features_for_parabola(x):
    return [x**2, x, 1]


def gradient_and_hessian(parameters, features):
    """Calculate the gradient and Hessian of the MSE."""
    number_of_parameters = len(parameters)
    gradient = [0.0] * number_of_parameters
    hessian = [
        [0.0] * number_of_parameters
        for _ in range(number_of_parameters)
    ]

    for x, y in DATA:
        features_at_x = features(x)
        prediction = sum(
            parameter * feature
            for parameter, feature in zip(parameters, features_at_x)
        )
        error = prediction - y

        for row in range(number_of_parameters):
            gradient[row] += (
                2 * error * features_at_x[row] / len(DATA)
            )
            for column in range(number_of_parameters):
                hessian[row][column] += (
                    2
                    * features_at_x[row]
                    * features_at_x[column]
                    / len(DATA)
                )

    return gradient, hessian


def solve_linear_system(matrix, vector):
    """Solve a small linear system for the Newton-Raphson update."""
    augmented = [
        matrix[row][:] + [vector[row]]
        for row in range(len(vector))
    ]
    size = len(vector)

    for pivot in range(size):
        pivot_row = max(
            range(pivot, size),
            key=lambda row: abs(augmented[row][pivot])
        )
        augmented[pivot], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot],
        )

        divisor = augmented[pivot][pivot]
        for column in range(pivot, size + 1):
            augmented[pivot][column] /= divisor

        for row in range(size):
            if row == pivot:
                continue
            multiplier = augmented[row][pivot]
            for column in range(pivot, size + 1):
                augmented[row][column] -= (
                    multiplier * augmented[pivot][column]
                )

    return [augmented[row][size] for row in range(size)]


def newton_raphson_multivariable(
    initial_parameters, features, tolerance=1e-7, max_iter=100
):
    """Minimize MSE with the multivariable Newton-Raphson method."""
    parameters = initial_parameters[:]
    iterations = [parameters[:]]

    for _ in range(max_iter):
        gradient, hessian = gradient_and_hessian(parameters, features)
        update = solve_linear_system(hessian, gradient)
        next_parameters = [
            parameter - change
            for parameter, change in zip(parameters, update)
        ]
        iterations.append(next_parameters[:])

        difference = max(
            abs(next_parameter - parameter)
            for next_parameter, parameter in zip(next_parameters, parameters)
        )
        if difference < tolerance:
            parameters = next_parameters
            break

        parameters = next_parameters

    return parameters, iterations


# ============================================================
# Plot numerical iterations
# ============================================================


def plot_fit(model, parameters, iterations, title, filename):
    x_values = [index / 20 for index in range(61)]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        [x for x, y in DATA],
        [y for x, y in DATA],
        color="black",
        label="Data points",
        zorder=3,
    )

    for index, iteration in enumerate(iterations[:-1]):
        y_values = [model(x, iteration) for x in x_values]
        plt.plot(
            x_values,
            y_values,
            "--",
            alpha=0.5,
            label=f"Newton iteration {index}",
        )

    final_y_values = [model(x, parameters) for x in x_values]
    plt.plot(
        x_values,
        final_y_values,
        color="tab:red",
        linewidth=2,
        label="Final fit",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ============================================================
# Run the numerical fits
# ============================================================


def print_result(name, parameters, model):
    print(name)
    print("Final parameters:", parameters)
    print("Final MSE:", mean_squared_error(parameters, model))
    print()


def main():
    media_folder = Path(__file__).parent / "media"

    line_parameters, line_iterations = newton_raphson_multivariable(
        [0.0, 0.0],
        features_for_line,
    )
    parabola_parameters, parabola_iterations = newton_raphson_multivariable(
        [0.0, 0.0, 0.0],
        features_for_parabola,
    )

    print_result("Numerical line fit: y = m*x + b", line_parameters, line)
    print_result(
        "Numerical parabola fit: y = a*x^2 + b*x + c",
        parabola_parameters,
        parabola,
    )

    plot_fit(
        line,
        line_parameters,
        line_iterations,
        "Numerical line fit: y = m*x + b",
        media_folder / "part2-line-fit.png",
    )
    plot_fit(
        parabola,
        parabola_parameters,
        parabola_iterations,
        "Numerical parabola fit: y = a*x^2 + b*x + c",
        media_folder / "part2-parabola-fit.png",
    )


if __name__ == "__main__":
    main()
