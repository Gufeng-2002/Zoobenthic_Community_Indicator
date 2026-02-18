"""RDA result containers — dataclasses only, no computation logic.

Mirrors ``models/pca.py`` (for PCA) and ``models/clustering.py`` (for Ward)
in purpose — holds every artefact produced by RDA fitting and testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ─── low-level fit containers ───────────────────────────────────────


@dataclass(frozen=True)
class PermutationTestResult:
    """Result of a single permutation test."""
    statistic: float
    p_value: float
    n_permutations: int
    null_distribution: np.ndarray


@dataclass(frozen=True)
class RDAScores:
    """Site, species, and (optional) biplot scores."""
    site_scores: pd.DataFrame
    species_scores: pd.DataFrame
    biplot_scores: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class RDAFit:
    """All quantities from a fitted RDA model."""
    X: pd.DataFrame
    Y: pd.DataFrame
    X_centered: pd.DataFrame
    Y_centered: pd.DataFrame
    coefficients: pd.DataFrame
    Y_hat: pd.DataFrame
    residuals: pd.DataFrame
    constrained_eigenvalues: pd.Series
    constrained_eigenvectors: pd.DataFrame
    explained_proportion: pd.Series
    cumulative_explained: pd.Series
    inertia_total: float
    inertia_constrained: float
    inertia_residual: float
    r2: float
    r2_adj: float
    df_model: int
    df_residual: int


# ─── pipeline-level result container ────────────────────────────────


@dataclass
class RDAResult:
    """Lightweight container returned by :func:`rda_pipeline`."""
    rda_model: Any                       # RDA (avoids circular import)
    scores: RDAScores
    global_test: PermutationTestResult
    axes_test: pd.DataFrame
    terms_test: pd.DataFrame
    axes_table: pd.DataFrame
    terms_table: pd.DataFrame
    env_data: pd.DataFrame
    taxa_data: pd.DataFrame
    cluster_labels: Optional[pd.Series] = None
    transformation_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        fit = self.rda_model.fit_
        return (
            f"RDAResult(n_sites={fit.X.shape[0]}, "
            f"n_env={fit.X.shape[1]}, n_taxa={fit.Y.shape[1]}, "
            f"R²={fit.r2:.4f}, adj-R²={fit.r2_adj:.4f}, "
            f"global_p={self.global_test.p_value:.4f})"
        )
