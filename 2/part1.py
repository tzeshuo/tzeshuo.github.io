"""SYDE 572 Assignment 2 — Part 1(a): polynomial basis regression.

Run this file from any directory with:
    python 2/part1.py

It uses the first 100 rows of ``housing.data`` for training and every
remaining row for testing, as required by the assignment.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path(__file__).parent / "assn2 material" / "assn2 material" / "housing.data"
TRAINING_SIZE = 100
DEGREES = range(1, 8)


def load_housing_data(data_path=DATA_PATH):
    """Return the 13 input features X and median house values y."""
    data = np.loadtxt(data_path)
    return data[:, :-1], data[:, -1]


def standardize_training_and_test_data(x_train, x_test):
    """Standardize using only training-set feature statistics.

    A zero standard deviation is replaced by one so a constant feature remains
    finite rather than causing a divide-by-zero error.
    """
    mean = x_train.mean(axis=0)
    standard_deviation = x_train.std(axis=0)
    standard_deviation[standard_deviation == 0] = 1.0
    return (
        (x_train - mean) / standard_deviation,
        (x_test - mean) / standard_deviation,
    )


def polynomial_design_matrix(x, degree):
    """Create [1, x_1, ..., x_d, x_1^2, ..., x_d^degree].

    This intentionally includes powers of individual variables only.  It does
    not include cross-terms such as x_1*x_2, matching the assignment wording.
    """
    if degree < 1:
        raise ValueError("Polynomial degree must be at least 1.")

    powers = [x**power for power in range(1, degree + 1)]
    return np.column_stack([np.ones(x.shape[0]), *powers])


def rms_error(predictions, targets):
    """Calculate root-mean-squared error."""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def polynomial_regression(x_train, y_train, x_test, y_test, degree):
    """Fit an unregularized polynomial basis model and return its results."""
    phi_train = polynomial_design_matrix(x_train, degree)
    phi_test = polynomial_design_matrix(x_test, degree)

    # np.linalg.lstsq solves the ordinary least-squares problem without adding
    # a regularization penalty.  It is more numerically stable than explicitly
    # forming and inverting (Phi.T @ Phi).
    weights, _, _, _ = np.linalg.lstsq(phi_train, y_train, rcond=None)

    train_error = rms_error(phi_train @ weights, y_train)
    test_error = rms_error(phi_test @ weights, y_test)
    return weights, train_error, test_error


def plot_errors(degrees, training_errors, test_errors, output_path):
    """Save the Part 1(a) RMS-error versus degree plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, training_errors, "o-", label="Training RMS error")
    plt.plot(degrees, test_errors, "s-", label="Test RMS error")
    plt.xlabel("Polynomial degree")
    plt.ylabel("RMS error (median house value, $1000s)")
    plt.title("Polynomial basis regression: training and test error")
    plt.xticks(list(degrees))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    x, y = load_housing_data()
    x_train, x_test = x[:TRAINING_SIZE], x[TRAINING_SIZE:]
    y_train, y_test = y[:TRAINING_SIZE], y[TRAINING_SIZE:]
    x_train, x_test = standardize_training_and_test_data(x_train, x_test)

    training_errors = []
    test_errors = []

    print("Part 1(a): unregularized polynomial basis regression")
    print("degree | training RMS | test RMS")
    print("-------+--------------+----------")
    for degree in DEGREES:
        _, train_error, test_error = polynomial_regression(
            x_train, y_train, x_test, y_test, degree
        )
        training_errors.append(train_error)
        test_errors.append(test_error)
        print(f"{degree:>6} | {train_error:>12.4f} | {test_error:>8.4f}")

    output_path = Path(__file__).parent / "part1a_polynomial_errors.png"
    plot_errors(DEGREES, training_errors, test_errors, output_path)
    print(f"\nSaved plot: {output_path}")


if __name__ == "__main__":
    main()
