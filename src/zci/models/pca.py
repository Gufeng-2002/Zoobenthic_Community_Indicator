"""PCA result container — dataclass only, no computation logic.

Holds every artefact produced by ``core.pca.run_pca``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

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

    def to_augmented_dataframe(
        self,
        pollution_score: pd.Series,
        selected_pcs: Sequence[str] | None = None,
        level0: str = "01_pollution_assessment",
        level1: str = "raw",
    ) -> pd.DataFrame:
        """Build a MultiIndex DataFrame with selected PC scores + composite score.

        Parameters
        ----------
        pollution_score : pd.Series
            Composite score per site (same index as ``self.scores``).
        selected_pcs : list of str, optional
            Which PCs to include.  *None* → all.
        level0 / level1 : str
            The first two levels of the column MultiIndex.

        Returns
        -------
        pd.DataFrame
            3-level column MultiIndex:
            ``(level0, level1, "PC1"), … (level0, level1, "Pollution_Score")``.
        """
        if selected_pcs is None:
            selected_pcs = list(self.scores.columns)

        # Combine selected PC scores + pollution score into a single plain DF
        parts = self.scores[list(selected_pcs)].copy()
        parts["Pollution_Score"] = pollution_score.values

        # Wrap into 3-level MultiIndex
        tuples = [(level0, level1, col) for col in parts.columns]
        parts.columns = pd.MultiIndex.from_tuples(
            tuples, names=["block", "subblock", "var"]
        )
        return parts
