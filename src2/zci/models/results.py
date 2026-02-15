"""Structured result containers — no logic, just shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    """Immutable container returned by ``core.pca.run_pca``.

    Attributes
    ----------
    loadings : pd.DataFrame
        *Scaled* loadings (variables × components).
        Each entry = eigenvector_jk × sqrt(eigenvalue_k).
    scores : pd.DataFrame
        Site scores in PC space (sites × components), after optional
        min-max standardisation.
    scores_raw : pd.DataFrame
        Site scores *before* any standardisation (sites × components).
    variance_info : pd.DataFrame
        Three rows (Explained Variance, Proportion, Cumulative)
        × n_components columns.
    n_components : int
        Number of retained components.
    """

    loadings: pd.DataFrame
    scores: pd.DataFrame
    scores_raw: pd.DataFrame
    variance_info: pd.DataFrame
    n_components: int

    # --- convenience helpers (no I/O, no side-effects) ----------------------

    def loadings_with_variance(self) -> pd.DataFrame:
        """Return a combined table: loadings rows + blank separator + variance rows.

        This is the "publication-ready" loadings table used by the original
        pipeline for the Excel export.
        """
        sep = pd.DataFrame(
            {col: [""] for col in self.loadings.columns},
            index=[""],
        ).astype(object)
        combined = pd.concat([self.loadings.astype(object), sep, self.variance_info.astype(object)])
        return combined
