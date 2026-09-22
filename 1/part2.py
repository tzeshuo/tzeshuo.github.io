"""Part 2: numerical fitting with multivariable Newton-Raphson."""

from pathlib import Path

import matplotlib.pyplot as plt


# The four data points from the assignment.
data = [(0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5)]
number_of_points = len(data)
tolerance = 1e-7
max_iterations = 100


# Models and MSE


def line(x, m, b):
    """Line model: y = m*x + b."""
    return m * x + b


def parabola(x, a, b, c):
    """Parabola model: y = a*x^2 + b*x + c."""
    return a * x**2 + b * x + c


def line_mse(m, b):
    """MSE for the line y = m*x + b."""
    total_error = 0.0

    for x, y in data:
        error = line(x, m, b) - y
        total_error += error**2

    return total_error / number_of_points


def parabola_mse(a, b, c):
    """MSE for the parabola y = a*x^2 + b*x + c."""
    total_error = 0.0

    for x, y in data:
        error = parabola(x, a, b, c) - y
        total_error += error**2

    return total_error / number_of_points


# Numerical line fit: y = m*x + b


def fit_line_newton():
    """Fit the line using a two-parameter Newton-Raphson update."""
    m = 0.0
    b = 0.0
    iterations = [(m, b)]

    # The Hessian is constant for a linear model, so calculate it once.
    hessian_mm = 2 * sum(x**2 for x, y in data) / number_of_points
    hessian_mb = 2 * sum(x for x, y in data) / number_of_points
    hessian_bb = 2.0

    determinant = hessian_mm * hessian_bb - hessian_mb**2

    for _ in range(max_iterations):
        gradient_m = 0.0
        gradient_b = 0.0

        for x, y in data:
            error = line(x, m, b) - y
            gradient_m += 2 * error * x / number_of_points
            gradient_b += 2 * error / number_of_points

        # Solve H * change = gradient for the two line parameters.
        change_m = (
            hessian_bb * gradient_m - hessian_mb * gradient_b
        ) / determinant
        change_b = (
            -hessian_mb * gradient_m + hessian_mm * gradient_b
        ) / determinant

        next_m = m - change_m
        next_b = b - change_b
        iterations.append((next_m, next_b))

        if max(abs(next_m - m), abs(next_b - b)) < tolerance:
            m = next_m
            b = next_b
            break

        m = next_m
        b = next_b

    return m, b, iterations


# ============================================================
# Numerical parabola fit: y = a*x^2 + b*x + c
# ============================================================


def fit_parabola_newton():
    """Fit the parabola using an explicit three-parameter Newton update."""
    a = 0.0
    b = 0.0
    c = 0.0
    iterations = [(a, b, c)]

    for _ in range(max_iterations):
        gradient_a = 0.0
        gradient_b = 0.0
        gradient_c = 0.0

        # Calculate the MSE gradient with respect to a, b, and c.
        for x, y in data:
            error = parabola(x, a, b, c) - y
            gradient_a += 2 * error * x**2 / number_of_points
            gradient_b += 2 * error * x / number_of_points
            gradient_c += 2 * error / number_of_points

        # Calculate the constant Hessian entries for the parabola model.
        hessian_aa = 2 * sum(x**4 for x, y in data) / number_of_points
        hessian_ab = 2 * sum(x**3 for x, y in data) / number_of_points
        hessian_ac = 2 * sum(x**2 for x, y in data) / number_of_points
        hessian_bb = 2 * sum(x**2 for x, y in data) / number_of_points
        hessian_bc = 2 * sum(x for x, y in data) / number_of_points
        hessian_cc = 2.0

        # Solve H * change = gradient using the explicit 3x3 determinant formula.
        determinant = (
            hessian_aa * (hessian_bb * hessian_cc - hessian_bc * hessian_bc)
            - hessian_ab * (hessian_ab * hessian_cc - hessian_bc * hessian_ac)
            + hessian_ac * (hessian_ab * hessian_bc - hessian_bb * hessian_ac)
        )

        determinant_a = (
            gradient_a * (hessian_bb * hessian_cc - hessian_bc * hessian_bc)
            - hessian_ab * (gradient_b * hessian_cc - hessian_bc * gradient_c)
            + hessian_ac * (gradient_b * hessian_bc - hessian_bb * gradient_c)
        )
        determinant_b = (
            hessian_aa * (gradient_b * hessian_cc - hessian_bc * gradient_c)
            - gradient_a * (hessian_ab * hessian_cc - hessian_bc * hessian_ac)
            + hessian_ac * (hessian_ab * gradient_c - gradient_b * hessian_ac)
        )
        determinant_c = (
            hessian_aa * (hessian_bb * gradient_c - gradient_b * hessian_bc)
            - hessian_ab * (hessian_ab * gradient_c - gradient_b * hessian_ac)
            + gradient_a * (hessian_ab * hessian_bc - hessian_bb * hessian_ac)
        )

        change_a = determinant_a / determinant
        change_b = determinant_b / determinant
        change_c = determinant_c / determinant

        next_a = a - change_a
        next_b = b - change_b
        next_c = c - change_c
        iterations.append((next_a, next_b, next_c))

        if max(
            abs(next_a - a),
            abs(next_b - b),
            abs(next_c - c),
        ) < tolerance:
            a = next_a
            b = next_b
            c = next_c
            break

        a = next_a
        b = next_b
        c = next_c

    return a, b, c, iterations


# ============================================================
# Plots of the intermediate Newton-Raphson fits
# ============================================================


def plot_line_fit(m, b, iterations, filename):
    x_values = [index / 20 for index in range(61)]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        [x for x, y in data],
        [y for x, y in data],
        color="black",
        label="Data points",
        zorder=3,
    )

    for index, (iteration_m, iteration_b) in enumerate(iterations[:-1]):
        y_values = [line(x, iteration_m, iteration_b) for x in x_values]
        plt.plot(
            x_values,
            y_values,
            "--",
            alpha=0.5,
            label=f"Newton iteration {index}",
        )

    final_y_values = [line(x, m, b) for x in x_values]
    plt.plot(x_values, final_y_values, color="tab:red", linewidth=2, label="Final fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Numerical line fit: y = m*x + b")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_parabola_fit(a, b, c, iterations, filename):
    x_values = [index / 20 for index in range(61)]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        [x for x, y in data],
        [y for x, y in data],
        color="black",
        label="Data points",
        zorder=3,
    )

    for index, (iteration_a, iteration_b, iteration_c) in enumerate(iterations[:-1]):
        y_values = [
            parabola(x, iteration_a, iteration_b, iteration_c)
            for x in x_values
        ]
        plt.plot(
            x_values,
            y_values,
            "--",
            alpha=0.5,
            label=f"Newton iteration {index}",
        )

    final_y_values = [parabola(x, a, b, c) for x in x_values]
    plt.plot(x_values, final_y_values, color="tab:red", linewidth=2, label="Final fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Numerical parabola fit: y = a*x^2 + b*x + c")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ============================================================
# Run the numerical methods
# ============================================================


def main():
    media_folder = Path(__file__).parent / "media"

    m, b, line_iterations = fit_line_newton()
    a, b_parabola, c, parabola_iterations = fit_parabola_newton()

    print("Numerical line fit: y = m*x + b")
    print(f"m = {m:.10f}")
    print(f"b = {b:.10f}")
    print(f"MSE = {line_mse(m, b):.10f}")
    print()

    print("Numerical parabola fit: y = a*x^2 + b*x + c")
    print(f"a = {a:.10f}")
    print(f"b = {b_parabola:.10f}")
    print(f"c = {c:.10f}")
    print(f"MSE = {parabola_mse(a, b_parabola, c):.10f}")

    plot_line_fit(
        m,
        b,
        line_iterations,
        media_folder / "part2-line-fit.png",
    )
    plot_parabola_fit(
        a,
        b_parabola,
        c,
        parabola_iterations,
        media_folder / "part2-parabola-fit.png",
    )


if __name__ == "__main__":
    main()
