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


def fit_single_polynomial_regression(x_train, y_train, x_test, y_test, degree):
    """Fit unregularized polynomial regression for a single degree."""
    phi_train = polynomial_design_matrix(x_train, degree)
    phi_test = polynomial_design_matrix(x_test, degree)
    weights = fit_linear_regression(phi_train, y_train)
    return weights, rms_error(phi_train @ weights, y_train), rms_error(phi_test @ weights, y_test)


def plot_part_a(train_errors, test_errors, degrees=DEGREES):
    """Plot Part 1(a) errors on a logarithmic y-scale so both train and test curves are visible."""
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_errors, "o-", label="Training RMS error")
    plt.plot(degrees, test_errors, "s-", label="Test RMS error")
    plt.yscale("log")
    plt.xlabel("Polynomial degree")
    plt.ylabel("RMS error (MEDV, $1000s, log scale)")
    plt.title("Part 1(a): Polynomial regression errors (degrees 1–7)")
    plt.xticks(list(degrees))
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "part1a_polynomial_errors.png", dpi=200)
    plt.close()


def polynomial_regression(x_train, y_train, x_test, y_test, degree=None):
    """Part 1(a) function as requested in assignment.

    If degree is None, fits degree 1 to 7 with training standardization, plots errors,
    and returns (train_errors, test_errors).
    If an integer degree is provided, fits for that degree on the provided inputs and
    returns (weights, train_error, test_error).
    """
    if degree is not None:
        return fit_single_polynomial_regression(x_train, y_train, x_test, y_test, degree)

    mean, std = fit_standardizer(x_train)
    x_train_std = standardize(x_train, mean, std)
    x_test_std = standardize(x_test, mean, std)
    train_errors, test_errors = [], []
    for deg in DEGREES:
        _, train_error, test_error = fit_single_polynomial_regression(
            x_train_std, y_train, x_test_std, y_test, deg
        )
        train_errors.append(train_error)
        test_errors.append(test_error)
    plot_part_a(train_errors, test_errors)
    return np.array(train_errors), np.array(test_errors)


def part_a(x_train, y_train, x_test, y_test):
    """Alias for Part 1(a) polynomial regression workflow."""
    return polynomial_regression(x_train, y_train, x_test, y_test)


def unstandardize_weights(weights, mean, std):
    """Convert standardized weights back to the original physical predictor scale."""
    w_unnorm = np.zeros_like(weights)
    w_unnorm[1:] = weights[1:] / std
    w_unnorm[0] = weights[0] - np.sum(weights[1:] * mean / std)
    return w_unnorm


def part_b(x_train, y_train):
    """Part 1(b): degree-one weights using training-set normalization.

    Returns:
        (std_weights, unstd_weights, mean, std)
    """
    mean, std = fit_standardizer(x_train)
    phi = polynomial_design_matrix(standardize(x_train, mean, std), 1)
    std_weights = fit_linear_regression(phi, y_train)
    unstd_weights = unstandardize_weights(std_weights, mean, std)
    return std_weights, unstd_weights, mean, std


def polynomial_regression_1d_vis(raw_train, y_train, raw_test, y_test, degree):
    """Part 1(c): use Boston's second feature (ZN) and plot its fitted curve."""
    mean, std = fit_standardizer(raw_train[:, None])
    train_x = standardize(raw_train[:, None], mean, std)
    test_x = standardize(raw_test[:, None], mean, std)
    weights, train_error, test_error = fit_single_polynomial_regression(train_x, y_train, test_x, y_test, degree)

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
    
    # Ensure full polynomial behavior is visible, including high-order oscillations
    y_min = min(0.0, float(curve_y.min()) - 5.0)
    y_max = max(55.0, float(curve_y.max()) + 5.0)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"part1c_degree_{degree}.png", dpi=200)
    plt.close()
    return train_error, test_error


def polynomial_regression_reg(raw_x, targets, degree=8, folds=10, plot=True):
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

    validation_errors = np.array(average_errors)
    if plot:
        plot_part_d(validation_errors)
    return validation_errors


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
    validation_errors = polynomial_regression_reg(x_train[:, 1], y_train, plot=True)
    return validation_errors, LAMBDA_VALUES[np.argmin(validation_errors)]


FEATURE_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x, y = load_housing_data()
    x_train, y_train = x[:TRAINING_SIZE], y[:TRAINING_SIZE]
    x_test, y_test = x[TRAINING_SIZE:], y[TRAINING_SIZE:]
    train_errors, test_errors = part_a(x_train, y_train, x_test, y_test)
    std_weights, unstd_weights, _, _ = part_b(x_train, y_train)
    one_d_errors = {
        degree: polynomial_regression_1d_vis(x_train[:, 1], y_train, x_test[:, 1], y_test, degree)
        for degree in (1, 3, 8)
    }
    validation_errors, best_lambda = part_d(x_train, y_train)

    print("=====================================================================")
    print("Part 1(a): Polynomial Regression Using All 13 Features")
    print("=====================================================================")
    print(f"{'Degree':>6} | {'Training RMS':>14} | {'Test RMS':>16}")
    print("-" * 42)
    for degree, train_error, test_error in zip(DEGREES, train_errors, test_errors):
        print(f"{degree:>6} | {train_error:>14.4f} | {test_error:>16.4f}")

    print("\n=====================================================================")
    print("Part 1(b): Degree-One Weights & Interpretation")
    print("=====================================================================")
    print(f"{'Weight':>8} | {'Feature':>10} | {'Std Weight':>12} | {'Unstd Weight':>14}")
    print("-" * 52)
    print(f"{'w0':>8} | {'Bias':>10} | {std_weights[0]:>12.6f} | {unstd_weights[0]:>14.6f}")
    for idx, name in enumerate(FEATURE_NAMES):
        w_label = f"w{idx+1} (x{idx})"
        print(f"{w_label:>8} | {name:>10} | {std_weights[idx+1]:>12.6f} | {unstd_weights[idx+1]:>14.6f}")

    print("\nSpecific Assignment Questions:")
    print(f"  - Weight for 6th feature RM (w5 in 0-based feature index / w6 with bias):")
    print(f"      Standardized:   {std_weights[6]:.6f}")
    print(f"      Unstandardized: {unstd_weights[6]:.6f} ($/room)")
    print(f"  - Weight for 8th feature DIS (w7 in 0-based feature index / w8 with bias):")
    print(f"      Standardized:   {std_weights[8]:.6f}")
    print(f"      Unstandardized: {unstd_weights[8]:.6f} ($/unit distance)")

    print("\n=====================================================================")
    print("Part 1(c): One-Dimensional Fits Using Feature 2 (ZN)")
    print("=====================================================================")
    for degree, (train_error, test_error) in one_d_errors.items():
        print(f"degree {degree}: training RMS = {train_error:.4f}, test RMS = {test_error:.4f}")

    print("\n=====================================================================")
    print("Part 1(d): Degree-Eight Regularized Regression, 10-Fold CV")
    print("=====================================================================")
    for regularization, validation_error in zip(LAMBDA_VALUES, validation_errors):
        print(f"lambda = {regularization:>7g}: average validation RMS = {validation_error:.4f}")
    print(f"Selected lambda: {best_lambda:g} (min validation RMS = {np.min(validation_errors):.4f})")
    print(f"\nAll plots generated and saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
