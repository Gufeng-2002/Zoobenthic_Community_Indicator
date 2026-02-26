"""Dataclasses for piecewise quantile regression results.

Lightweight containers — no computation here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ClusterQRResult:
    """Aggregated piecewise-QR results for one cluster."""

    cluster_id: int
    n_sites: int
    n_taxa: int              # informational (not used in QR directly)

    # Multi-quantile results ─────────────────────────────────────────
    # from fit_multi_quantile: {tau → PiecewiseQRResult}
    quantile_results: Dict[float, Any]

    # Sensitivity analysis ───────────────────────────────────────────
    # {frac → SubsampleResult}
    sensitivity_results: Optional[Dict[float, Any]] = None
    true_params: Optional[np.ndarray] = None
    sensitivity_tau: Optional[float] = None

    # Param names (same for all taus)
    param_names: List[str] = field(default_factory=list)

    # ── Summary helpers ──────────────────────────────────────────────

    @property
    def taus(self) -> List[float]:
        return sorted(self.quantile_results.keys())

    def median_breakpoint(self, tau: float = 0.50) -> float:
        """Return the breakpoint estimate at the given quantile level."""
        res = self.quantile_results.get(tau)
        if res is None:
            return np.nan
        return float(res.breakpoints[0])

    def summary_line(self) -> str:
        """One-line summary for logging."""
        bp50 = self.median_breakpoint(0.50)
        return (
            f"Cluster {self.cluster_id}: "
            f"{self.n_sites} sites, "
            f"{len(self.taus)} quantile levels, "
            f"median breakpoint = {bp50:.3f}"
        )


@dataclass
class PQRPipelineResult:
    """Top-level result container for the piecewise-QR pipeline."""

    cluster_results: Dict[int, ClusterQRResult]
    clusters: List[int]
    x_name: str     # e.g. "Pollution_Score"
    y_name: str     # e.g. "ZCI"

    def summary(self) -> str:
        lines = [
            "Piecewise Quantile Regression Pipeline",
            "=" * 50,
            f"  predictor: {self.x_name}",
            f"  response:  {self.y_name}",
        ]
        for cl in self.clusters:
            lines.append(f"  {self.cluster_results[cl].summary_line()}")
        return "\n".join(lines)
