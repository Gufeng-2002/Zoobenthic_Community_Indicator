"""LDA result containers — dataclasses only, no computation logic.

Mirrors ``models/rda.py`` and ``models/clustering.py`` in purpose —
holds every artefact produced by the LDA classification pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ─── single-fit result ──────────────────────────────────────────────


@dataclass
class LDAFit:
    """Quantities from a single full-data LDA fit."""

    model: Any                              # sklearn LinearDiscriminantAnalysis
    scaler: Any                             # sklearn StandardScaler (or None)
    env_data: pd.DataFrame                  # z-scored env data used for fitting
    cluster_labels: np.ndarray              # true labels
    predictions: np.ndarray                 # predicted labels
    accuracy: float
    confusion_matrix: np.ndarray            # (k × k)
    classification_report: Dict[str, Any]   # sklearn dict
    explained_variance_ratio: np.ndarray    # per LD axis
    env_variables: List[str]
    cluster_names: List[str]


# ─── Wilks' Lambda variable importance ──────────────────────────────


@dataclass
class WilksImportance:
    """Per-variable importance via Wilks' Lambda drop-one."""

    axes_summary: pd.DataFrame              # LD axes explained %
    variable_importance: pd.DataFrame       # sorted by F-statistic
    wilks_lambda_full: float
    overall_significance: Dict[str, float]  # chi_square, df, p_value
    significance_dict: Dict[str, Dict]      # var → {p_value, significance}


# ─── Monte Carlo cross-validation result ────────────────────────────


@dataclass
class MCCVResult:
    """Aggregated results from Monte Carlo Cross-Validation."""

    mean_accuracy: float
    std_accuracy: float
    median_accuracy: float
    min_accuracy: float
    max_accuracy: float
    aggregate_confusion_matrix: np.ndarray
    avg_classification_report: Dict[str, Dict[str, float]]
    cluster_names: List[str]
    n_iterations: int
    test_size: float
    all_true_labels: List
    all_predictions: List


# ─── pipeline-level result container ────────────────────────────────


@dataclass
class LDAResult:
    """Lightweight container returned by :func:`lda_pipeline`."""

    lda_fit: LDAFit
    wilks: WilksImportance
    mccv: MCCVResult
    env_significance_table: pd.DataFrame    # publication-ready table
    nonref_predictions: Optional[pd.Series] = None
    nonref_probabilities: Optional[pd.DataFrame] = None
    transformation_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        n_ref = len(self.lda_fit.cluster_labels)
        acc = self.lda_fit.accuracy
        mccv_acc = self.mccv.mean_accuracy
        return (
            f"LDAResult(n_ref={n_ref}, accuracy={acc:.2%}, "
            f"MCCV={mccv_acc:.2%}±{self.mccv.std_accuracy:.2%})"
        )
