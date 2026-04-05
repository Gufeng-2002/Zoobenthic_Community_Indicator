"""Hazard (HZD) scoring — McPhedran-style TEC/PEC hazard score.

Implements the hazard score approach described by McPhedran et al.
Each chemical is mapped to a toxicity effect percentage using a
chemical-specific sigmoidal curve calibrated so that:

* ``TEC`` corresponds to 5% toxicity
* ``PEC`` corresponds to 50% toxicity

The paper-specific HZD score then applies two additional rules:

* any concentration below ``TEC`` contributes 0% toxicity
* the summed site score is capped at 100%

Public API
----------
load_tec_pec_benchmarks
    Read the TEC/PEC lookup table from Excel.
compute_hzd_curve_parameters
    Solve the per-chemical sigmoid coefficients from TEC/PEC anchors.
compute_hzd_effect_matrix
    Compute per-site, per-chemical effect percentages.
compute_hzd_score
    Compute the capped site-level HZD toxicity score.
classify_hzd
    Classify sites using the HZD toxicity-scale anchors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

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


def compute_hzd_curve_parameters(
    benchmarks: pd.DataFrame,
    *,
    tec_effect_percent: float = 5.0,
    pec_effect_percent: float = 50.0,
) -> pd.DataFrame:
    """Solve the McPhedran sigmoid parameters for each chemical.

    The paper defines the chemical-specific effect curve as:

    ``Effect(%) = 100 / (1 + A * exp(-k * C))``

    where ``A`` and ``k`` are chosen so that the curve passes through
    ``(TEC, tec_effect_percent)`` and ``(PEC, pec_effect_percent)``.
    """
    required_cols = {"TEC", "PEC"}
    missing_cols = required_cols.difference(benchmarks.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Benchmark table is missing required columns: {missing}")

    if not (0 < tec_effect_percent < pec_effect_percent < 100):
        raise ValueError(
            "Expected 0 < tec_effect_percent < pec_effect_percent < 100."
        )

    tec = pd.to_numeric(benchmarks["TEC"], errors="coerce")
    pec = pd.to_numeric(benchmarks["PEC"], errors="coerce")

    invalid = tec.isna() | pec.isna() | (tec <= 0) | (pec <= tec)
    if invalid.any():
        bad = ", ".join(map(str, benchmarks.index[invalid].tolist()))
        raise ValueError(
            "TEC/PEC benchmark rows must have positive TEC values and PEC > TEC. "
            f"Invalid chemicals: {bad}"
        )

    tec_ratio = 100.0 / tec_effect_percent - 1.0
    pec_ratio = 100.0 / pec_effect_percent - 1.0

    curve_k = np.log(tec_ratio / pec_ratio) / (pec - tec)
    curve_A = pec_ratio * np.exp(curve_k * pec)

    return pd.DataFrame(
        {
            "curve_A": curve_A,
            "curve_k": curve_k,
        },
        index=benchmarks.index,
    )


def compute_hzd_effect_matrix(
    chemical_data: pd.DataFrame,
    benchmarks: pd.DataFrame,
    *,
    tec_effect_percent: float = 5.0,
    pec_effect_percent: float = 50.0,
    zero_below_tec: bool = True,
) -> pd.DataFrame:
    """Compute per-site, per-chemical HZD effect percentages.

    Parameters
    ----------
    chemical_data : pd.DataFrame
        Sites × chemicals. Column names must match the benchmark index.
    benchmarks : pd.DataFrame
        Must contain ``TEC`` and ``PEC`` columns indexed by chemical name.
    tec_effect_percent, pec_effect_percent : float
        Toxicity percentages assigned to the TEC and PEC anchors.
    zero_below_tec : bool
        If ``True``, chemical concentrations below ``TEC`` contribute 0%.

    Returns
    -------
    pd.DataFrame
        Sites × chemicals effect percentages on the 0–100 scale.
    """
    common = [c for c in benchmarks.index if c in chemical_data.columns]
    if not common:
        raise ValueError(
            "No overlapping chemicals between data columns and benchmark index."
        )

    chem_subset = chemical_data[common].apply(pd.to_numeric, errors="coerce")
    bench_subset = benchmarks.loc[common]
    curve_params = compute_hzd_curve_parameters(
        bench_subset,
        tec_effect_percent=tec_effect_percent,
        pec_effect_percent=pec_effect_percent,
    )

    pec_ratio = 100.0 / pec_effect_percent - 1.0
    exp_arg = chem_subset.rsub(bench_subset["PEC"], axis=1).mul(
        curve_params["curve_k"], axis=1,
    )

    effect = 100.0 / (1.0 + pec_ratio * np.exp(np.clip(exp_arg, -700, 700)))
    effect = effect.where(chem_subset.notna())

    if zero_below_tec:
        effect = effect.mask(chem_subset.lt(bench_subset["TEC"], axis=1), 0.0)

    return effect.clip(lower=0.0, upper=100.0)


def compute_hzd_score(
    chemical_data: pd.DataFrame,
    benchmarks: pd.DataFrame,
    *,
    quotient_type: str | None = None,
    tec_effect_percent: float = 5.0,
    pec_effect_percent: float = 50.0,
    zero_below_tec: bool = True,
    cap_percent: float = 100.0,
) -> pd.Series:
    """Compute the capped site-level HZD toxicity score.

    Parameters
    ----------
    chemical_data : pd.DataFrame
        Sites × chemicals. Column names must match the benchmark index.
    benchmarks : pd.DataFrame
        Must have ``TEC`` and ``PEC`` columns indexed by chemical name.
    quotient_type : {"PEC", "TEC"} or None
        Retained for backward compatibility with the previous quotient-based
        implementation. The HZD method always uses both TEC and PEC.

    Returns
    -------
    pd.Series
        Per-site HZD toxicity score on the 0–100 scale.
    """
    if quotient_type not in (None, "PEC", "TEC"):
        raise ValueError(
            f"quotient_type must be None, 'PEC', or 'TEC', got {quotient_type!r}"
        )
    if cap_percent <= 0:
        raise ValueError(f"cap_percent must be positive, got {cap_percent!r}")

    effect_matrix = compute_hzd_effect_matrix(
        chemical_data,
        benchmarks,
        tec_effect_percent=tec_effect_percent,
        pec_effect_percent=pec_effect_percent,
        zero_below_tec=zero_below_tec,
    )

    hzd_score = effect_matrix.sum(axis=1, min_count=1).clip(
        lower=0.0,
        upper=cap_percent,
    )
    hzd_score.name = "HZD_Toxicity_Percent"
    return hzd_score


def classify_hzd(
    hzd_score: pd.Series,
    thresholds: Dict[str, float] | None = None,
) -> pd.Series:
    """Classify sites using the HZD toxicity-scale anchors.

    Default thresholds align with the paper's score interpretation:

    * ``0``      → ``"Zero"``
    * ``0–5``    → ``"Below TEC"``
    * ``5–50``   → ``"TEC-PEC"``
    * ``> 50``   → ``"Above PEC"``

    Parameters
    ----------
    hzd_score : pd.Series
        HZD toxicity score per site on the 0–100 scale.
    thresholds : dict or None
        Custom upper breakpoints for the post-zero bins.

    Returns
    -------
    pd.Series
        Categorical labels, same index as *hzd_score*.
    """
    if thresholds is None:
        thresholds = {"Below TEC": 0.0, "TEC-PEC": 5.0, "Above PEC": 50.0}

    bins = [-np.inf] + sorted(thresholds.values()) + [np.inf]
    labels = ["Zero"] + sorted(thresholds, key=thresholds.get)

    return pd.cut(hzd_score, bins=bins, labels=labels, include_lowest=True)
