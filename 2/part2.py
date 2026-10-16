"""SYDE 572 Assignment 2 — Part 2: Gaussian basis regression.

Run with ``python 2/part2.py``. The script creates every Part 2(e–f)
figure in ``2/media`` and prints the numerical results for the report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path(__file__).parent / "assn2 material" / "assn2 material" / "housing.data"
OUTPUT_DIR = Path(__file__).parent / "media"
TRAINING_SIZE = 100
K_VALUES = list(range(5, 105, 10))  # [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
LAMBDA_VALUES = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
DEFAULT_SCALE = 2.0  # Gaussian basis function width s = 2


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


def rms_error(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def fit_linear_regression(phi, targets, regularization=0.0):
    """Fit least squares with an optional L2 penalty; bias is not penalized."""
    if regularization == 0:
        return np.linalg.lstsq(phi, targets, rcond=None)[0]
    penalty = np.eye(phi.shape[1]) * regularization
    penalty[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + penalty, phi.T @ targets)


def gaussian_design_matrix(x, centers, scale=DEFAULT_SCALE):
    """Return design matrix [1, phi_1(x), ..., phi_K(x)] using Gaussian basis functions.

    Each basis function is:
        phi_j(x) = exp(- ||x - mu_j||^2 / (2 * scale^2))
    """
    x = np.asarray(x)
    centers = np.asarray(centers)
    # Pairwise squared Euclidean distances between points and centers
    dist_sq = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    phi = np.exp(-dist_sq / (2.0 * scale ** 2))
    return np.column_stack([np.ones(x.shape[0]), phi])


def plot_part_e(k_values, train_errors, test_errors):
    """Plot Part 2(e) training and test RMS errors versus number of basis functions."""
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, train_errors, "o-", label="Training RMS error")
    plt.plot(k_values, test_errors, "s-", label="Test RMS error")
    plt.xlabel("Number of Gaussian basis functions (K)")
    plt.ylabel("RMS error (MEDV, $1000s)")
    plt.title("Part 2(e): Gaussian basis regression errors (s = 2)")
    plt.xticks(list(k_values))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "part2e_gaussian_errors.png", dpi=200)
    plt.close()


def gaussian_regression(x_train, y_train, x_test, y_test, k_values=None, scale=DEFAULT_SCALE, seed=42, plot=True):
    """Part 2(e): unregularized Gaussian basis regression across varying K.

    Uses the first 100 points as training data and remainder as testing data.
    Generates a random permutation of training points and takes the first K as centers.
    """
    if k_values is None:
        k_values = K_VALUES

    # Normalize predictor variables using training-set statistics
    mean, std = fit_standardizer(x_train)
    x_train_std = standardize(x_train, mean, std)
    x_test_std = standardize(x_test, mean, std)

    # Random permutation of training points to pick centers
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_train_std))

    train_errors, test_errors = [], []
    for k in k_values:
        centers = x_train_std[perm[:k]]
        phi_train = gaussian_design_matrix(x_train_std, centers, scale=scale)
        phi_test = gaussian_design_matrix(x_test_std, centers, scale=scale)
        weights = fit_linear_regression(phi_train, y_train)

        train_errors.append(rms_error(phi_train @ weights, y_train))
        test_errors.append(rms_error(phi_test @ weights, y_test))

    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)

    if plot:
        plot_part_e(k_values, train_errors, test_errors)

    return train_errors, test_errors


def plot_part_f(validation_errors, lambda_values=LAMBDA_VALUES):
    """Plot Part 2(f) cross-validation errors versus regularization constant lambda."""
    positive = lambda_values > 0
    plt.figure(figsize=(8, 5))
    plt.semilogx(
        lambda_values[positive],
        validation_errors[positive],
        "o-",
        label="L2-regularized (90 basis functions)",
    )
    plt.axhline(
        validation_errors[0],
        color="tab:gray",
        linestyle="--",
        label=f"Unregularized baseline (λ=0: {validation_errors[0]:.2f})",
    )
    plt.xlabel("Regularization constant λ (log scale)")
    plt.ylabel("Average 10-fold validation RMS error")
    plt.title("Part 2(f): Gaussian basis regression regularization (90 basis functions)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "part2f_cross_validation.png", dpi=200)
    plt.close()


def gaussian_regression_reg(x_train, y_train, num_basis=90, lambda_values=None, folds=10, scale=DEFAULT_SCALE, seed=42, plot=True):
    """Part 2(f): L2-regularized Gaussian basis regression using 10-fold CV.

    Uses all 13 features of the first 100 points with 90 basis functions.
    In each fold, standardizer is learned from the fold's training partition.
    """
    if lambda_values is None:
        lambda_values = LAMBDA_VALUES

    rng = np.random.default_rng(seed)
    fold_indices = np.array_split(np.arange(len(y_train)), folds)

    # Precompute fold design matrices and partitions once so all lambda values
    # are evaluated on the exact same basis functions
    fold_partitions = []
    for validation_indices in fold_indices:
        training_mask = np.ones(len(y_train), dtype=bool)
        training_mask[validation_indices] = False
        fold_x_train, fold_y_train = x_train[training_mask], y_train[training_mask]
        fold_x_val, fold_y_val = x_train[validation_indices], y_train[validation_indices]

        # Fit scaling on this fold's training partition only
        mean, std = fit_standardizer(fold_x_train)
        fold_x_train_std = standardize(fold_x_train, mean, std)
        fold_x_val_std = standardize(fold_x_val, mean, std)

        # Centers: choose 90 centers from the fold's 90 training points
        fold_perm = rng.permutation(len(fold_x_train_std))
        centers = fold_x_train_std[fold_perm[:num_basis]]

        phi_train = gaussian_design_matrix(fold_x_train_std, centers, scale=scale)
        phi_val = gaussian_design_matrix(fold_x_val_std, centers, scale=scale)

        fold_partitions.append((phi_train, fold_y_train, phi_val, fold_y_val))

    average_errors = []
    for regularization in lambda_values:
        fold_errors = []
        for phi_train, fold_y_train, phi_val, fold_y_val in fold_partitions:
            weights = fit_linear_regression(phi_train, fold_y_train, regularization)
            fold_errors.append(rms_error(phi_val @ weights, fold_y_val))
        average_errors.append(np.mean(fold_errors))

    validation_errors = np.array(average_errors)
    if plot:
        plot_part_f(validation_errors, lambda_values)

    best_lambda = lambda_values[np.argmin(validation_errors)]
    return validation_errors, best_lambda


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x, y = load_housing_data()
    x_train, y_train = x[:TRAINING_SIZE], y[:TRAINING_SIZE]
    x_test, y_test = x[TRAINING_SIZE:], y[TRAINING_SIZE:]

    print("=====================================================================")
    print("Part 2(e): Gaussian Basis Regression (No Regularization, s = 2)")
    print("=====================================================================")
    train_errors, test_errors = gaussian_regression(x_train, y_train, x_test, y_test)
    print(f"{'Basis (K)':>10} | {'Training RMS':>14} | {'Test RMS':>14}")
    print("-" * 44)
    for k, train_err, test_err in zip(K_VALUES, train_errors, test_errors):
        print(f"{k:>10} | {train_err:>14.4f} | {test_err:>14.4f}")

    print("\n=====================================================================")
    print("Part 2(f): Gaussian Basis Regularization (90 Basis Functions, 10-Fold CV)")
    print("=====================================================================")
    validation_errors, best_lambda = gaussian_regression_reg(x_train, y_train)
    for regularization, validation_error in zip(LAMBDA_VALUES, validation_errors):
        print(f"lambda = {regularization:>7g}: average validation RMS = {validation_error:.4f}")
    print(f"\nSelected lambda from CV: {best_lambda:g} (min validation RMS = {np.min(validation_errors):.4f})")
    print(f"Plots saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

