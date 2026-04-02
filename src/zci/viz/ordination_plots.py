"""PCA ordination biplot of reference sites in environmental space."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Cluster marker shapes (up to 10 clusters)
_CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "p", "h", "*"]


def plot_env_pca_ordination(
    env_ref: pd.DataFrame,
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    *,
    env_feature_names: Sequence[str] | None = None,
    arrow_scale: float = 1.0,
    figsize: tuple[float, float] = (9, 8),
    dpi: int = 180,
) -> tuple[plt.Figure, plt.Axes]:
    """Create PCA biplot of reference sites coloured by prediction accuracy.

    Parameters
    ----------
    env_ref : DataFrame
        Environmental features for reference sites (rows=sites, cols=variables).
    true_labels : Series
        Ward cluster labels (index = StationID).
    predicted_labels : Series
        MRT-predicted cluster labels (index = StationID).
    env_feature_names : list[str] or None
        Display names for loading arrows.  Defaults to column names.
    arrow_scale : float
        Extra multiplier on loading arrows.
    """
    common = env_ref.index.intersection(true_labels.index).intersection(predicted_labels.index)
    env = env_ref.loc[common]
    true = true_labels.loc[common].astype(int)
    pred = predicted_labels.loc[common].astype(int)

    if env_feature_names is None:
        env_feature_names = list(env.columns)

    # ---- standardise & PCA ------------------------------------------------
    scaler = StandardScaler()
    X_std = scaler.fit_transform(env.values)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_std)  # (n, 2)

    eigenvalues = pca.explained_variance_   # length 2
    loadings = pca.components_.T            # (p, 2)
    # Scale loadings by sqrt(eigenvalue) for biplot convention
    scaled_loadings = loadings * np.sqrt(eigenvalues)[np.newaxis, :] * arrow_scale

    var_explained = pca.explained_variance_ratio_ * 100

    # ---- correct / incorrect ----------------------------------------------
    correct = (true.values == pred.values)

    # ---- plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    unique_clusters = sorted(true.unique())
    for cl in unique_clusters:
        marker = _CLUSTER_MARKERS[(cl - 1) % len(_CLUSTER_MARKERS)]

        mask_correct = (true.values == cl) & correct
        mask_wrong = (true.values == cl) & ~correct

        if mask_correct.any():
            ax.scatter(
                scores[mask_correct, 0], scores[mask_correct, 1],
                marker=marker, s=70, edgecolors="black", linewidths=0.6,
                c="#2E7D32", label=f"Cluster {cl} (correct)",
                zorder=3,
            )
        if mask_wrong.any():
            ax.scatter(
                scores[mask_wrong, 0], scores[mask_wrong, 1],
                marker=marker, s=70, edgecolors="black", linewidths=0.6,
                c="#D62728", label=f"Cluster {cl} (misclassified)",
                zorder=3,
            )

    # Site labels
    for i, sid in enumerate(common):
        ax.annotate(
            str(sid), (scores[i, 0], scores[i, 1]),
            textcoords="offset points", xytext=(4, 4),
            fontsize=6, color="#555555", zorder=2,
        )

    # ---- loading arrows ---------------------------------------------------
    for j, fname in enumerate(env_feature_names):
        dx, dy = scaled_loadings[j, 0], scaled_loadings[j, 1]
        ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
            zorder=4,
        )
        ax.text(
            dx * 1.08, dy * 1.08, fname,
            fontsize=9, fontweight="bold", color="#1f77b4",
            ha="center", va="center", zorder=5,
        )

    # ---- axes & legend ----------------------------------------------------
    ax.axhline(0, color="grey", linewidth=0.5, zorder=0)
    ax.axvline(0, color="grey", linewidth=0.5, zorder=0)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)", fontsize=12)
    ax.set_title(
        "PCA Ordination of Reference Sites in Environmental Space",
        fontsize=13, fontweight="bold", pad=12,
    )

    n_correct = int(correct.sum())
    n_total = len(correct)
    ax.text(
        0.02, 0.02,
        f"Correct: {n_correct}/{n_total} ({100 * n_correct / n_total:.1f}%)",
        transform=ax.transAxes, fontsize=10, color="#4B5563",
        va="bottom", ha="left",
    )

    ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, 1.0),
        frameon=True, framealpha=0.9, fontsize=9,
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()

    return fig, ax


def save_env_pca_ordination(
    env_ref: pd.DataFrame,
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    output_path: str | Path,
    **kwargs,
) -> Path:
    """Create and save the PCA ordination biplot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_env_pca_ordination(env_ref, true_labels, predicted_labels, **kwargs)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
