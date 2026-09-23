
import math
from pathlib import Path

import matplotlib.pyplot as plt


def newton_raphson(
    x0, y0, f, df, ddf, initial_guess=0.0, tolerance=1e-7, max_iter=100
):
    x = initial_guess
    iterations = [x]

    for _ in range(max_iter):
        # D'(x)
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)

        # D''(x)
        D_double_prime = 2 + 2 * df(x) ** 2 + 2 * (f(x) - y0) * ddf(x)

        # Newton-Raphson update step
        next_x = x - D_prime / D_double_prime
        iterations.append(next_x)

        if abs(next_x - x) < tolerance:
            x = next_x
            break

        x = next_x

    shortest_distance = ((x - x0) ** 2 + (f(x) - y0) ** 2) ** 0.5
    return shortest_distance, x, iterations


def golden_section_search(x0, y0, f, a, b, tolerance=1e-7):
    """Minimize the squared distance on [a, b] without derivatives."""
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi

    def dist_sq(x):
        return (x - x0) ** 2 + (f(x) - y0) ** 2

    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)
    iterations = [(a, b, x1, x2)]

    while abs(b - a) > tolerance:
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + resphi * (b - a)
            f_x1 = dist_sq(x1)
        else:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = b - resphi * (b - a)
            f_x2 = dist_sq(x2)
        iterations.append((a, b, x1, x2))

    best_x = (a + b) / 2
    return math.sqrt(dist_sq(best_x)), best_x, iterations


def _sample_values(start, end, count=500):
    step = (end - start) / (count - 1)
    return [start + index * step for index in range(count)]


def plot_newton_raphson(x0, y0, f, iterations, x_limits, title, filename):
    x_values = _sample_values(*x_limits)
    y_values = [f(value) for value in x_values]
    iteration_y = [f(value) for value in iterations]
    closest_x = iterations[-1]
    closest_y = f(closest_x)
    distance = math.hypot(closest_x - x0, closest_y - y0)
    update_count = len(iterations) - 1

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="Function")
    plt.scatter(x0, y0, color="black", zorder=3, label="Given point")
    plt.plot(
        [x0, closest_x],
        [y0, closest_y],
        color="tab:blue",
        linewidth=2,
        label=f"Shortest distance = {distance:.4f}",
    )
    plt.scatter(
        iterations,
        iteration_y,
        color="tab:red",
        zorder=3,
        label=f"Newton iterates ({update_count} updates)",
    )
    plt.plot(iterations, iteration_y, "--", color="tab:red", alpha=0.7)
    plt.scatter(
        closest_x,
        closest_y,
        color="tab:green",
        edgecolor="black",
        zorder=4,
        label="Closest point",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Newton-Raphson: {title}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_golden_section(x0, y0, f, iterations, x_limits, title, filename):
    x_values = _sample_values(*x_limits)
    y_values = [f(value) for value in x_values]
    left, right = iterations[-1][:2]
    best_x = (left + right) / 2
    best_y = f(best_x)
    distance = math.hypot(best_x - x0, best_y - y0)
    update_count = len(iterations) - 1
    evaluation_x = [value for item in iterations for value in item[2:]]
    evaluation_y = [f(value) for value in evaluation_x]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="Function")
    plt.scatter(x0, y0, color="black", zorder=3, label="Given point")
    plt.plot(
        [x0, best_x],
        [y0, best_y],
        color="tab:blue",
        linewidth=2,
        label=f"Shortest distance = {distance:.4f}",
    )
    plt.scatter(
        evaluation_x,
        evaluation_y,
        color="tab:purple",
        alpha=0.35,
        s=18,
        zorder=2,
        label=f"Golden-section evaluations ({update_count} updates)",
    )
    plt.scatter(
        best_x,
        best_y,
        color="tab:green",
        edgecolor="black",
        zorder=4,
        label="Closest point",
    )

    # Show the first three and final search intervals as orange regions.
    interval_indices = [0, 1, 2, len(iterations) - 1]
    interval_labels = ["1st search", "2nd search", "3rd search", "Final"]
    interval_alphas = [0.08, 0.12, 0.16, 0.22]
    axis = plt.gca()

    for index, label, alpha in zip(
        interval_indices, interval_labels, interval_alphas
    ):
        left, right = iterations[index][:2]
        interval_label = "Search intervals" if index == 0 else None
        plt.axvspan(
            left,
            right,
            color="tab:orange",
            alpha=alpha,
            label=interval_label,
        )
        axis.text(
            (left + right) / 2,
            0.93 - 0.07 * interval_labels.index(label),
            label,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#7a4300",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Golden-section search: {title}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def parabola(x):
    return x**2 + 5


def parabola_derivative(x):
    return 2 * x


def parabola_second_derivative(x):
    return 2


def exponential(x):
    return math.exp(x)


def exponential_derivative(x):
    return math.exp(x)


def exponential_second_derivative(x):
    return math.exp(x)


def logarithmic(x):
    return math.log(x)


def logarithmic_derivative(x):
    return 1 / x


def logarithmic_second_derivative(x):
    return -1 / x**2


def print_result(method, function_name, point, result, function):
    distance, closest_x = result[:2]
    iteration_count = len(result[2]) - 1
    print(f"{function_name} - {method}")
    print(f"  Given point: {point}")
    print(f"  Closest point: ({closest_x:.10f}, {function(closest_x):.10f})")
    print(f"  Shortest distance: {distance:.10f}\n")
    print(f"  Iterations: {iteration_count}\n")


def run_case(
    function_name, function, derivative, second_derivative, point,
    initial_guess, interval, plot_limits, plot_name=None, make_plots=False
):
    x0, y0 = point
    newton_result = newton_raphson(
        x0, y0, function, derivative, second_derivative, initial_guess
    )
    golden_result = golden_section_search(x0, y0, function, *interval)

    print_result("Newton-Raphson", function_name, point, newton_result, function)
    print_result("Golden-section", function_name, point, golden_result, function)

    if make_plots:
        media_folder = Path(__file__).parent / "media"
        plot_newton_raphson(
            x0, y0, function, newton_result[2], plot_limits, function_name,
            media_folder / f"{plot_name}-newton.png"
        )
        plot_golden_section(
            x0, y0, function, golden_result[2], plot_limits, function_name,
            media_folder / f"{plot_name}-golden-section.png"
        )


def main():
    points = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]
    print("PART 1: y = x^2 + 5\n")

    for point_index, point in enumerate(points, start=1):
        run_case(
            "y = x^2 + 5", parabola, parabola_derivative,
            parabola_second_derivative, point, 0.0, (-10, 10), (-10, 10),
            plot_name=f"parabola-point-{point_index}", make_plots=True
        )

    # The positive interval for log(x) keeps every evaluation in its domain.
    run_case(
        "y = exp(x)", exponential, exponential_derivative,
        exponential_second_derivative, (2, 0), 0.0, (-10, 10), (-3, 3),
        plot_name="exponential", make_plots=True
    )
    run_case(
        "y = log(x)", logarithmic, logarithmic_derivative,
        logarithmic_second_derivative, (2, 0), 1.0, (0.1, 5), (0.1, 5),
        plot_name="logarithmic", make_plots=True
    )

    plt.show()


if __name__ == "__main__":
    main()
