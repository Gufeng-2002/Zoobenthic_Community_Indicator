"""Hazard (HZD) scoring — consensus-based TEC / PEC quotients.

Implements the mean PEC-quotient method from MacDonald et al. (2000):

    *Development and Evaluation of Consensus-Based Sediment Quality
    Guidelines for Freshwater Ecosystems.*

For each site *i* the mean PEC quotient is:

    mean_PEC_Q_i = (1/n) * Σ_j (C_ij / PEC_j)

where *C_ij* is the measured concentration of chemical *j* at site *i*,
*PEC_j* is the consensus-based probable-effect concentration, and *n*
is the number of chemicals with non-missing data at that site.

Public API
----------
load_tec_pec_benchmarks
    Read the TEC/PEC lookup table from Excel.
compute_hzd_score
    Compute per-site mean PEC quotient from raw chemical concentrations.
classify_hzd
    Classify sites into hazard categories based on mean PEC quotient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


# ─── default benchmark path (relative to project data/) ─────────────

_DEFAULT_BENCHMARK_PATH = "data/TEC_PEC_consensus/10_chemicals_TEC_PEC.xlsx"


def load_tec_pec_benchmarks(
    path: str | Path,
) -> pd.DataFrame:
    """Read the TEC/PEC benchmark table.

    Expected columns: ``chemical name``, ``TEC``, ``PEC``, ``unit``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``chemical name`` with columns ``TEC``, ``PEC``, ``unit``.
    """
    df = pd.read_excel(Path(path))
    df = df.set_index("chemical name")
    return df


def compute_hzd_score(
    chemical_data: pd.DataFrame,
    benchmarks: pd.DataFrame,
    *,
    quotient_type: str = "PEC",
) -> pd.Series:
    """Compute per-site mean quotient (HZD score).

    Parameters
    ----------
    chemical_data : pd.DataFrame
        Sites × chemicals.  Column names must match the benchmark index.
    benchmarks : pd.DataFrame
        Must have a ``PEC`` (and optionally ``TEC``) column, indexed by
        chemical name.
    quotient_type : {"PEC", "TEC"}
        Which benchmark to divide by.

    Returns
    -------
    pd.Series
        Per-site mean quotient, same index as *chemical_data*.
    """
    if quotient_type not in ("PEC", "TEC"):
        raise ValueError(f"quotient_type must be 'PEC' or 'TEC', got {quotient_type!r}")

    # Identify chemicals present in both data and benchmarks
    common = [c for c in benchmarks.index if c in chemical_data.columns]
    if not common:
        raise ValueError(
            "No overlapping chemicals between data columns and benchmark index."
        )

    bench_values = benchmarks.loc[common, quotient_type]
    chem_subset = chemical_data[common]

    # Quotient matrix: C_ij / benchmark_j
    quotients = chem_subset.div(bench_values, axis=1)

    # Mean across chemicals, ignoring NaN
    mean_q = quotients.mean(axis=1)
    mean_q.name = f"mean_{quotient_type}_quotient"

    return mean_q


def classify_hzd(
    hzd_score: pd.Series,
    thresholds: Dict[str, float] | None = None,
) -> pd.Series:
    """Classify sites into hazard categories.

    Default thresholds follow MacDonald et al. (2000):

    * ``< 0.1``  → "Minimal"
    * ``0.1–0.5`` → "Low"
    * ``0.5–1.0`` → "Moderate"
    * ``> 1.0``  → "High"

    Parameters
    ----------
    hzd_score : pd.Series
        Mean PEC quotient per site.
    thresholds : dict or None
        Custom breakpoints ``{"Low": 0.1, "Moderate": 0.5, "High": 1.0}``.

    Returns
    -------
    pd.Series
        Categorical labels, same index as *hzd_score*.
    """
    if thresholds is None:
        thresholds = {"Low": 0.1, "Moderate": 0.5, "High": 1.0}

    bins = [-np.inf] + sorted(thresholds.values()) + [np.inf]
    labels = ["Minimal"] + sorted(thresholds, key=thresholds.get)

    return pd.cut(hzd_score, bins=bins, labels=labels)
