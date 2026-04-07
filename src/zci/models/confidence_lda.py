"""Confidence-aware LDA result containers — dataclasses only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .lda import LDAFit, MCCVResult, WilksImportance


@dataclass
class ConfidenceLDAModelResult:
    """Result container for a single confidence-aware LDA model."""

    model_name: str
    training_subset: str                    # e.g. "Core", "Core+Peripheral"
    n_train: int
    lda_fit: LDAFit
    wilks: WilksImportance
    mccv: MCCVResult

    # Per-site evaluation on training sites
    train_eval: pd.DataFrame                # Site | Original_Cluster | Status |
                                            # Predicted_Cluster | Prob_C1..C3 |
                                            # p_max | delta_p | Role(Train)

    # Per-site evaluation on held-out sites (may be empty)
    heldout_eval: Optional[pd.DataFrame] = None

    def cv_accuracy(self) -> float:
        return self.mccv.mean_accuracy

    def cv_accuracy_std(self) -> float:
        return self.mccv.std_accuracy

    def train_accuracy(self) -> float:
        return self.lda_fit.accuracy

    def summary(self) -> str:
        ho = ""
        if self.heldout_eval is not None and len(self.heldout_eval) > 0:
            ho = f", heldout={len(self.heldout_eval)}"
        return (
            f"{self.model_name}(n_train={self.n_train}, "
            f"train_acc={self.train_accuracy():.2%}, "
            f"CV={self.cv_accuracy():.2%}±{self.cv_accuracy_std():.2%}"
            f"{ho})"
        )


@dataclass
class ConfidenceLDAComparison:
    """Comparison across multiple confidence-aware LDA models."""

    models: Dict[str, ConfidenceLDAModelResult]
    weights: pd.Series                       # confidence weight w_i per site
    comparison_table: pd.DataFrame           # final site-level comparison
    summary_table: pd.DataFrame              # model-level summary metrics

    def summary(self) -> str:
        lines = ["Confidence-Aware LDA Comparison:"]
        for name, m in self.models.items():
            lines.append(f"  {m.summary()}")
        return "\n".join(lines)
