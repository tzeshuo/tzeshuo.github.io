from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

# Set seed for reproducible jitter and random operations
np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent / "media"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
DATA_PATH = Path(__file__).parent / "assn3 material" / "iris.data"


class IrisDataset:
    def __init__(self, data, target, feature_names, target_names):
        self.data = np.asarray(data, dtype=float)
        self.target = np.asarray(target, dtype=int)
        self.feature_names = feature_names
        self.target_names = target_names


def load_iris_data():
    """Load the Iris dataset from local UCI cache, UCI repository URL, or sklearn fallback."""
    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
    target_names = ["setosa", "versicolor", "virginica"]
    class_map = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }

    lines = []
    # 1. Check for locally cached UCI iris.data
    if DATA_PATH.exists():
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    else:
        # 2. Attempt download from UCI archive
        import urllib.request
        try:
            req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8")
                lines = [l.strip() for l in content.splitlines() if l.strip()]
        except Exception:
            pass

    if lines:
        rows = [l.split(",") for l in lines]
        X = np.array([[float(v) for v in r[:4]] for r in rows], dtype=float)
        y = np.array([class_map[r[4]] for r in rows], dtype=int)
        assert len(X) == 150 and X.shape[1] == 4, f"Unexpected shape {X.shape}"
        return IrisDataset(X, y, feature_names, target_names)

    # 3. Fallback to scikit-learn dataset
    from sklearn.datasets import load_iris
    return load_iris()

# Set consistent matplotlib styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight"
})


# ==============================================================================
# Part 1: PCA by Hand Verification & Plot
# ==============================================================================
def run_part1():
    print("=" * 70)
    print("Part 1: PCA by Hand")
    print("=" * 70)

    # 5 points with 2 features
    X = np.array([
        [-2, -1],
        [-1,  0],
        [ 0,  1],
        [ 1,  2],
        [ 2,  3]
    ], dtype=float)

    mu = np.mean(X, axis=0)
    X_centered = X - mu
    print(f"Dataset X:\n{X}")
    print(f"Mean vector mu: {mu}")
    print(f"Centered X:\n{X_centered}")

    # Scatter matrix S = X_centered.T @ X_centered
    S = X_centered.T @ X_centered
    # Sample covariance matrix C (N=5)
    C_sample = S / (len(X) - 1)  # using N-1=4
    print(f"Scatter matrix (X_c^T X_c):\n{S}")
    print(f"Sample covariance (N-1=4):\n{C_sample}")

    # Eigenvalues and eigenvectors of S
    eigvals, eigvecs = np.linalg.eigh(S)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    w1 = eigvecs[:, 0]
    # Standardize sign so first entry is positive
    if w1[0] < 0:
        w1 = -w1

    print(f"Eigenvalues: {eigvals}")
    print(f"First Principal Component w1: {w1} (analytical: [1/sqrt(2), 1/sqrt(2)] = {1/np.sqrt(2):.6f})")

    # Q2: Reconstruction error of (3, 3)
    p2 = np.array([3.0, 3.0])
    p2_c = p2 - mu
    z2 = np.dot(p2_c, w1)
    p2_recon_c = z2 * w1
    p2_recon = p2_recon_c + mu
    err2_vec = p2 - p2_recon
    err2_l2 = np.linalg.norm(err2_vec)

    print(f"\nPoint (3, 3):")
    print(f"  Centered: {p2_c}")
    print(f"  Scalar coordinate z1: {z2:.6f}")
    print(f"  Reconstructed: {p2_recon}")
    print(f"  Error vector: {err2_vec}")
    print(f"  L2 reconstruction error: {err2_l2:.6f} (analytical: 1/sqrt(2) = {1/np.sqrt(2):.6f})")

    # Q3: Reconstruction error of (3, 4)
    p3 = np.array([3.0, 4.0])
    p3_c = p3 - mu
    z3 = np.dot(p3_c, w1)
    p3_recon_c = z3 * w1
    p3_recon = p3_recon_c + mu
    err3_vec = p3 - p3_recon
    err3_l2 = np.linalg.norm(err3_vec)

    print(f"\nPoint (3, 4):")
    print(f"  Centered: {p3_c}")
    print(f"  Scalar coordinate z1: {z3:.6f}")
    print(f"  Reconstructed: {p3_recon}")
    print(f"  Error vector: {err3_vec}")
    print(f"  L2 reconstruction error: {err3_l2:.6f} (analytical: 0.0)")

    # Plot Part 1 verification
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[:, 0], X[:, 1], color="#2b5c8f", s=70, zorder=4, label="Training Points (5)")
    ax.scatter(mu[0], mu[1], color="#d95f02", s=100, marker="X", zorder=5, label=f"Mean $\\mu = ({mu[0]:.0f}, {mu[1]:.0f})$")

    # Line of PC1
    t = np.linspace(-4, 5, 100)
    line_pts = mu + t[:, None] * w1
    ax.plot(line_pts[:, 0], line_pts[:, 1], color="#e7298a", linestyle="--", linewidth=2, label="1st Principal Component Axis")

    # Plot test point (3,3) and projection
    ax.scatter(p2[0], p2[1], color="#7570b3", s=90, marker="o", zorder=6, label="Test point (3,3)")
    ax.scatter(p2_recon[0], p2_recon[1], color="#7570b3", s=70, marker="s", facecolors="none", linewidth=2, zorder=6, label="Projection of (3,3)")
    ax.plot([p2[0], p2_recon[0]], [p2[1], p2_recon[1]], color="#7570b3", linestyle=":", linewidth=2, label=r"Error $\|e\|_2 = 1/\sqrt{2} \approx 0.707$")

    # Plot test point (3,4)
    ax.scatter(p3[0], p3[1], color="#1b9e77", s=100, marker="*", zorder=6, label="Test point (3,4) [on PC1, err = 0]")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Part 1: Analytical PCA, First Principal Component & Projections")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "part1_pca_hand_plot.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved Part 1 plot to: {fig_path}")


# ==============================================================================
# Part 2: Data Visualization (Marginalization)
# ==============================================================================
def run_part2(iris):
    print("\n" + "=" * 70)
    print("Part 2: Data Visualization (Marginalization)")
    print("=" * 70)

    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # 1D Marginalization: Petal length (feature 2)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    feat_idx = 2  # petal length
    for c_idx, (c_name, color) in enumerate(zip(target_names, colors)):
        vals = X[y == c_idx, feat_idx]
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(vals, np.full_like(vals, c_idx) + jitter, color=color, alpha=0.7, s=40, label=c_name)
        # Plot mean marker
        ax.scatter(vals.mean(), c_idx, color="black", marker="|", s=250, linewidth=2.5, zorder=5)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(target_names)
    ax.set_xlabel(f"{feature_names[feat_idx]} (cm)")
    ax.set_title("Part 2: 1D Marginalization onto Petal Length (Feature 3)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig1d_path = OUTPUT_DIR / "part2_1d_petal_length.png"
    plt.savefig(fig1d_path)
    plt.close()
    print(f"Saved Part 2 1D plot to: {fig1d_path}")

    # 2D Marginalization: Petal length vs Petal width (features 2 and 3)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    f_x, f_y = 2, 3  # petal length, petal width
    for c_idx, (c_name, color) in enumerate(zip(target_names, colors)):
        ax.scatter(X[y == c_idx, f_x], X[y == c_idx, f_y], color=color, alpha=0.8, s=45, label=c_name)

    ax.set_xlabel(f"{feature_names[f_x]} (cm)")
    ax.set_ylabel(f"{feature_names[f_y]} (cm)")
    ax.set_title("Part 2: 2D Marginalization onto Petal Length vs. Petal Width")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig2d_path = OUTPUT_DIR / "part2_2d_petal_length_vs_width.png"
    plt.savefig(fig2d_path)
    plt.close()
    print(f"Saved Part 2 2D plot to: {fig2d_path}")

    # Comparison 2D: Sepal length vs Sepal width (showing poor separation)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    f_x2, f_y2 = 0, 1
    for c_idx, (c_name, color) in enumerate(zip(target_names, colors)):
        ax.scatter(X[y == c_idx, f_x2], X[y == c_idx, f_y2], color=color, alpha=0.8, s=45, label=c_name)

    ax.set_xlabel(f"{feature_names[f_x2]} (cm)")
    ax.set_ylabel(f"{feature_names[f_y2]} (cm)")
    ax.set_title("Part 2 Comparison: 2D Marginalization onto Sepal Length vs. Sepal Width")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig2d_sepal_path = OUTPUT_DIR / "part2_2d_sepal_length_vs_width.png"
    plt.savefig(fig2d_sepal_path)
    plt.close()
    print(f"Saved Part 2 Sepal 2D plot to: {fig2d_sepal_path}")

    # 3D Marginalization: Petal length, Petal width, Sepal length (features 2, 3, 0)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    f_z = 0  # sepal length
    for c_idx, (c_name, color) in enumerate(zip(target_names, colors)):
        ax.scatter(
            X[y == c_idx, f_x],
            X[y == c_idx, f_y],
            X[y == c_idx, f_z],
            color=color,
            alpha=0.8,
            s=40,
            label=c_name
        )

    ax.set_xlabel(f"{feature_names[f_x]} (cm)", labelpad=9)
    ax.set_ylabel(f"{feature_names[f_y]} (cm)", labelpad=9)
    ax.set_zlabel(f"{feature_names[f_z]} (cm)", labelpad=9)
    ax.set_title("Part 2: 3D Marginalization (Petal Length, Petal Width, Sepal Length)")
    ax.view_init(elev=20, azim=125)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig3d_path = OUTPUT_DIR / "part2_3d_features.png"
    plt.savefig(fig3d_path)
    plt.close()
    print(f"Saved Part 2 3D plot to: {fig3d_path}")


# ==============================================================================
# Part 3: PCA on Iris
# ==============================================================================
def run_part3(iris):
    print("\n" + "=" * 70)
    print("Part 3: PCA on Iris")
    print("=" * 70)

    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)

    print("Principal Component Vectors (Eigenvectors / Axes):")
    for i, comp in enumerate(pca.components_):
        print(f"  PC{i+1}: {comp} | norm = {np.linalg.norm(comp):.4f}")
        for f_name, weight in zip(feature_names, comp):
            print(f"      {f_name:>20}: {weight:+.4f}")

    print("\nExplained Variance:")
    for i, (var, ratio) in enumerate(zip(pca.explained_variance_, pca.explained_variance_ratio_)):
        print(f"  PC{i+1}: Variance = {var:.4f}, Explained Ratio = {ratio*100:.2f}% (Cumulative = {np.sum(pca.explained_variance_ratio_[:i+1])*100:.2f}%)")

    # 1D Visualization: PC1 (Major Axis)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c_idx, (c_name, color) in enumerate(zip(target_names, colors)):
        vals = X_pca[y == c_idx, 0]
        jitter = np.random.normal(0, 0.05, size=len(vals))
        ax.scatter(vals, np.full_like(vals, c_idx) + jitter, color=color, alpha=0.75, s=45, label=c_name)
        ax.scatter(vals.mean(), c_idx, color="black", marker="|", s=250, linewidth=2.5, zorder=5)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(target_names)
    ax.set_xlabel(f"First Principal Component (PC1) - {pca.explained_variance_ratio_[0]*100:.1f}% Variance")
    ax.set_title("Part 3: Data Marginalized onto Major Axis (PC1)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig1d_path = OUTPUT_DIR / "part3_pca_1d.png"
    plt.savefig(fig1d_path)
    plt.close()
    print(f"Saved Part 3 1D plot to: {fig1d_path}")

    # 2D Visualization: Top 2 Axes (PC1 vs PC2)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c_idx, (c_name, color) in enumerate(zip(target_names, colors)):
        ax.scatter(
            X_pca[y == c_idx, 0],
            X_pca[y == c_idx, 1],
            color=color,
            alpha=0.8,
            s=45,
            label=c_name
        )

    var2d = (pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]) * 100
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
    ax.set_title(f"Part 3: Top 2 PCA Axes (Explains {var2d:.2f}% Total Variance)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig2d_path = OUTPUT_DIR / "part3_pca_2d.png"
    plt.savefig(fig2d_path)
    plt.close()
    print(f"Saved Part 3 2D plot to: {fig2d_path}")

    # 3D Visualization: Top 3 Axes (PC1, PC2, PC3)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for c_idx, (c_name, color) in enumerate(zip(target_names, colors)):
        ax.scatter(
            X_pca[y == c_idx, 0],
            X_pca[y == c_idx, 1],
            X_pca[y == c_idx, 2],
            color=color,
            alpha=0.8,
            s=40,
            label=c_name
        )

    var3d = np.sum(pca.explained_variance_ratio_[:3]) * 100
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", labelpad=9)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", labelpad=9)
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)", labelpad=9)
    ax.set_title(f"Part 3: Top 3 PCA Axes (Explains {var3d:.2f}% Total Variance)")
    ax.view_init(elev=20, azim=130)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig3d_path = OUTPUT_DIR / "part3_pca_3d.png"
    plt.savefig(fig3d_path)
    plt.close()
    print(f"Saved Part 3 3D plot to: {fig3d_path}")

    return pca, X_pca


# ==============================================================================
# Part 4: Data Compression & Frobenius Norm Connection
# ==============================================================================
def run_part4(iris, pca):
    print("\n" + "=" * 70)
    print("Part 4: Data Compression & Frobenius Norm Analysis")
    print("=" * 70)

    X = iris.data
    N, D = X.shape
    X_centered = X - np.mean(X, axis=0)

    # Frobenius norm squared of centered data matrix
    frob_norm_sq = np.sum(X_centered ** 2)
    # Trace of scatter matrix
    scatter_matrix = X_centered.T @ X_centered
    trace_scatter = np.trace(scatter_matrix)

    print(f"Centered matrix Frobenius norm squared: ||X_c||_F^2 = {frob_norm_sq:.6f}")
    print(f"Trace of scatter matrix Tr(X_c^T X_c)       = {trace_scatter:.6f}")
    print(f"Sum of eigenvalues sum(lambda_j)          = {np.sum(pca.explained_variance_) * (N - 1):.6f}")

    # Variance explained by 2D representation
    var_pc1 = pca.explained_variance_ratio_[0]
    var_pc2 = pca.explained_variance_ratio_[1]
    var_2d = var_pc1 + var_pc2
    print(f"\nQuestion 1: Variance explained by 2D representation:")
    print(f"  PC1: {var_pc1 * 100:.4f}%")
    print(f"  PC2: {var_pc2 * 100:.4f}%")
    print(f"  Total 2D Explained Variance: {var_2d * 100:.4f}%")

    # Question 2: PCs required for 50%
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    num_pcs_50 = np.argmax(cum_var >= 0.50) + 1
    print(f"\nQuestion 2: PCs required for 50% variance:")
    print(f"  Cumulative ratios: {[f'{x*100:.2f}%' for x in cum_var]}")
    print(f"  Required PCs: {num_pcs_50} (PC1 alone explains {var_pc1 * 100:.2f}% > 50%)")

    # Standardized comparison (correlation matrix)
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(X)
    pca_std = PCA().fit(X_std)
    std_var_pc1 = pca_std.explained_variance_ratio_[0]
    std_var_pc2 = pca_std.explained_variance_ratio_[1]
    std_cum_var = np.cumsum(pca_std.explained_variance_ratio_)
    print(f"\nComparison: Standardized Features (Unit Variance / Correlation Matrix):")
    print(f"  PC1: {std_var_pc1*100:.2f}%, PC2: {std_var_pc2*100:.2f}%")
    print(f"  Total 2D Explained Variance: {(std_var_pc1 + std_var_pc2)*100:.2f}%")
    print(f"  PCs for 50%: {np.argmax(std_cum_var >= 0.50) + 1} (PC1 explains {std_var_pc1*100:.2f}% > 50%)")

    # Scree Plot
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    components = np.arange(1, D + 1)
    bars = ax1.bar(components, pca.explained_variance_ratio_ * 100, color="#4575b4", alpha=0.75, label="Individual % Variance")
    ax1.set_xlabel("Principal Component Index")
    ax1.set_ylabel("Explained Variance Ratio (%)", color="#4575b4")
    ax1.set_xticks(components)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(components, cum_var * 100, color="#d73027", marker="o", linewidth=2.2, label="Cumulative % Variance")
    ax2.axhline(50, color="gray", linestyle="--", linewidth=1.5, label="50% Threshold")
    ax2.set_ylabel("Cumulative Explained Variance (%)", color="#d73027")
    ax2.set_ylim(0, 105)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Part 4: Scree Plot & Cumulative Explained Variance on Iris")
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "part4_scree_variance.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved Part 4 Scree plot to: {fig_path}")


# ==============================================================================
# Part 5: Non-linear Components (ISOMAP vs PCA)
# ==============================================================================
def run_part5(iris, pca, X_pca):
    print("\n" + "=" * 70)
    print("Part 5: Non-linear Components (ISOMAP with k=3)")
    print("=" * 70)

    X = iris.data
    y = iris.target
    target_names = iris.target_names
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # ISOMAP with k=3 nearest neighbors as requested
    isomap = Isomap(n_neighbors=3, n_components=2)
    X_isomap = isomap.fit_transform(X)

    # Plot ISOMAP alone
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c_idx, (c_name, color) in enumerate(zip(target_names, colors)):
        ax.scatter(
            X_isomap[y == c_idx, 0],
            X_isomap[y == c_idx, 1],
            color=color,
            alpha=0.8,
            s=45,
            label=c_name
        )

    ax.set_xlabel("ISOMAP Component 1")
    ax.set_ylabel("ISOMAP Component 2")
    ax.set_title("Part 5: ISOMAP Embedding (3-Nearest Neighbours, 2 Components)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig_isomap_path = OUTPUT_DIR / "part5_isomap_k3.png"
    plt.savefig(fig_isomap_path)
    plt.close()
    print(f"Saved Part 5 ISOMAP plot to: {fig_isomap_path}")

    # Side-by-side comparison: PCA vs ISOMAP
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for c_idx, (c_name, color) in enumerate(zip(target_names, colors)):
        ax1.scatter(
            X_pca[y == c_idx, 0],
            X_pca[y == c_idx, 1],
            color=color,
            alpha=0.8,
            s=40,
            label=c_name
        )
    ax1.set_xlabel(f"PCA Axis 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)")
    ax1.set_ylabel(f"PCA Axis 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)")
    ax1.set_title("PCA: Linear Orthogonal Projection")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="upper right")

    for c_idx, (c_name, color) in enumerate(zip(target_names, colors)):
        ax2.scatter(
            X_isomap[y == c_idx, 0],
            X_isomap[y == c_idx, 1],
            color=color,
            alpha=0.8,
            s=40,
            label=c_name
        )
    ax2.set_xlabel("ISOMAP Component 1")
    ax2.set_ylabel("ISOMAP Component 2")
    ax2.set_title("ISOMAP: Geodesic Manifold Embedding (k=3)")
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="upper right")

    plt.suptitle("Part 5 Comparison: Linear PCA vs. Non-linear ISOMAP (k=3)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig_comp_path = OUTPUT_DIR / "part5_pca_vs_isomap.png"
    plt.savefig(fig_comp_path)
    plt.close()
    print(f"Saved Part 5 Comparison plot to: {fig_comp_path}")


def main():
    iris = load_iris_data()
    run_part1()
    run_part2(iris)
    pca, X_pca = run_part3(iris)
    run_part4(iris, pca)
    run_part5(iris, pca, X_pca)
    print("\nAll Part 1–5 calculations and plots completed successfully!")


if __name__ == "__main__":
    main()

