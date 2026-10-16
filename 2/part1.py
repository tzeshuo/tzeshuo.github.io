"""SYDE 572 Assignment 2 — Part 1: polynomial basis regression.

Run with ``python 2/part1.py``. The script creates every Part 1(a–d)
figure in ``2/media`` and prints the numerical results for the report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path(__file__).parent / "assn2 material" / "assn2 material" / "housing.data"
OUTPUT_DIR = Path(__file__).parent / "media"
TRAINING_SIZE = 100
DEGREES = range(1, 8)
LAMBDA_VALUES = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])


def load_housing_data(data_path=DATA_PATH):
    """Return the 13 predictor columns X and house values y."""
    data = np.loadtxt(data_path)
    return data[:, :-1], data[:, -1]


def fit_standardizer(x):
    """Return training-set statistics, protecting constant columns."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return mean, np.where(std == 0, 1.0, std)


def standardize(x, mean, std):
    return (x - mean) / std


def polynomial_design_matrix(x, degree):
    """Return [1, x_1, ..., x_d, x_1², ..., x_d^degree], without cross-terms."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if degree < 1:
        raise ValueError("Polynomial degree must be at least 1.")
    return np.column_stack([np.ones(x.shape[0]), *(x**p for p in range(1, degree + 1))])


def rms_error(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def fit_linear_regression(phi, targets, regularization=0.0):
    """Fit least squares with an optional L2 penalty; bias is not penalized."""
    if regularization == 0:
        return np.linalg.lstsq(phi, targets, rcond=None)[0]
    penalty = np.eye(phi.shape[1]) * regularization
    penalty[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + penalty, phi.T @ targets)


def polynomial_regression(x_train, y_train, x_test, y_test, degree):
    """Fit unregularized polynomial regression and return weights and RMS errors."""
    phi_train = polynomial_design_matrix(x_train, degree)
    phi_test = polynomial_design_matrix(x_test, degree)
    weights = fit_linear_regression(phi_train, y_train)
    return weights, rms_error(phi_train @ weights, y_train), rms_error(phi_test @ weights, y_test)


def plot_part_a(train_errors, test_errors):
    plt.figure(figsize=(8, 5))
    plt.plot(DEGREES, train_errors, "o-", label="Training RMS error")
    plt.plot(DEGREES, test_errors, "s-", label="Test RMS error")
    plt.xlabel("Polynomial degree")
    plt.ylabel("RMS error (MEDV, $1000s)")
    plt.title("Part 1(a): Polynomial regression errors")
    plt.xticks(list(DEGREES))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "part1a_polynomial_errors.png", dpi=200)
    plt.close()


def part_a(x_train, y_train, x_test, y_test):
    """Part 1(a): degree 1–7 errors using every feature."""
    mean, std = fit_standardizer(x_train)
    x_train = standardize(x_train, mean, std)
    x_test = standardize(x_test, mean, std)
    train_errors, test_errors = [], []
    for degree in DEGREES:
        _, train_error, test_error = polynomial_regression(x_train, y_train, x_test, y_test, degree)
        train_errors.append(train_error)
        test_errors.append(test_error)
    plot_part_a(train_errors, test_errors)
    return np.array(train_errors), np.array(test_errors)


def part_b(x_train, y_train):
    """Part 1(b): degree-one weights using training-set normalization."""
    mean, std = fit_standardizer(x_train)
    phi = polynomial_design_matrix(standardize(x_train, mean, std), 1)
    return fit_linear_regression(phi, y_train)


def polynomial_regression_1d_vis(raw_train, y_train, raw_test, y_test, degree):
    """Part 1(c): use Boston's second feature (ZN) and plot its fitted curve."""
    mean, std = fit_standardizer(raw_train[:, None])
    train_x = standardize(raw_train[:, None], mean, std)
    test_x = standardize(raw_test[:, None], mean, std)
    weights, train_error, test_error = polynomial_regression(train_x, y_train, test_x, y_test, degree)

    curve_x = np.linspace(min(raw_train.min(), raw_test.min()), max(raw_train.max(), raw_test.max()), 1000)
    curve_x_normalized = standardize(curve_x[:, None], mean, std)
    curve_y = polynomial_design_matrix(curve_x_normalized, degree) @ weights
    plt.figure(figsize=(8, 5))
    plt.scatter(raw_train, y_train, s=28, alpha=0.8, label="Training data")
    plt.plot(curve_x, curve_y, color="tab:red", linewidth=2, label=f"Degree {degree} polynomial")
    plt.scatter(raw_test, y_test, s=22, alpha=0.55, label="Test data")
    plt.xlabel("ZN: proportion zoned for lots over 25,000 sq. ft.")
    plt.ylabel("MEDV ($1000s)")
    plt.title(f"Part 1(c): One-dimensional polynomial regression (degree {degree})")
    # Keep the housing-value observations readable when high-degree extrapolation
    # produces values far outside the target range.
    plt.ylim(0, 60)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"part1c_degree_{degree}.png", dpi=200)
    plt.close()
    return train_error, test_error


def polynomial_regression_reg(raw_x, targets, degree=8, folds=10):
    """Return 10-fold average validation RMS errors for every L2 constant.

    Each fold learns its scaling only from the fold's training partition.
    """
    fold_indices = np.array_split(np.arange(len(targets)), folds)
    average_errors = []
    for regularization in LAMBDA_VALUES:
        fold_errors = []
        for validation_indices in fold_indices:
            training_mask = np.ones(len(targets), dtype=bool)
            training_mask[validation_indices] = False
            fold_x_train, fold_y_train = raw_x[training_mask], targets[training_mask]
            fold_x_validation, fold_y_validation = raw_x[validation_indices], targets[validation_indices]
            mean, std = fit_standardizer(fold_x_train[:, None])
            phi_train = polynomial_design_matrix(standardize(fold_x_train[:, None], mean, std), degree)
            phi_validation = polynomial_design_matrix(standardize(fold_x_validation[:, None], mean, std), degree)
            weights = fit_linear_regression(phi_train, fold_y_train, regularization)
            fold_errors.append(rms_error(phi_validation @ weights, fold_y_validation))
        average_errors.append(np.mean(fold_errors))
    return np.array(average_errors)


def plot_part_d(validation_errors):
    positive = LAMBDA_VALUES > 0
    plt.figure(figsize=(8, 5))
    plt.semilogx(
        LAMBDA_VALUES[positive],
        validation_errors[positive],
        "o-",
        label=f"L2-regularized (λ=0 baseline: {validation_errors[0]:.1f})",
    )
    plt.xlabel("Regularization constant λ (log scale)")
    plt.ylabel("Average 10-fold validation RMS error")
    plt.title("Part 1(d): Degree-8 polynomial regularization")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "part1d_cross_validation.png", dpi=200)
    plt.close()


def part_d(x_train, y_train):
    """Part 1(d): degree-eight, 10-fold CV using only the second feature (ZN)."""
    validation_errors = polynomial_regression_reg(x_train[:, 1], y_train)
    plot_part_d(validation_errors)
    return validation_errors, LAMBDA_VALUES[np.argmin(validation_errors)]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x, y = load_housing_data()
    x_train, y_train = x[:TRAINING_SIZE], y[:TRAINING_SIZE]
    x_test, y_test = x[TRAINING_SIZE:], y[TRAINING_SIZE:]
    train_errors, test_errors = part_a(x_train, y_train, x_test, y_test)
    weights = part_b(x_train, y_train)
    one_d_errors = {
        degree: polynomial_regression_1d_vis(x_train[:, 1], y_train, x_test[:, 1], y_test, degree)
        for degree in (1, 3, 8)
    }
    validation_errors, best_lambda = part_d(x_train, y_train)

    print("Part 1(a): polynomial regression using all 13 features")
    print("degree | training RMS | test RMS")
    for degree, train_error, test_error in zip(DEGREES, train_errors, test_errors):
        print(f"{degree:>6} | {train_error:>12.4f} | {test_error:>8.4f}")
    print("\nPart 1(b): degree-one weights (bias followed by data columns)")
    for index, weight in enumerate(weights):
        print(f"{'bias' if index == 0 else f'feature {index}':>10}: {weight:.6f}")
    print("\nPart 1(c): one-dimensional fits using feature 2 (ZN)")
    for degree, (train_error, test_error) in one_d_errors.items():
        print(f"degree {degree}: training RMS = {train_error:.4f}, test RMS = {test_error:.4f}")
    print("\nPart 1(d): degree-eight regularized regression, 10-fold CV")
    for regularization, validation_error in zip(LAMBDA_VALUES, validation_errors):
        print(f"lambda = {regularization:>7g}: average validation RMS = {validation_error:.4f}")
    print(f"Selected lambda: {best_lambda:g}")
    print(f"\nPlots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
