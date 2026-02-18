"""One-way ANOVA across cluster groups — pure computation, no plotting.

Produces publication-ready summary tables with SS, df, F, p, cluster
mean ± SEM columns, sample-size footer, and significance legend.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import f_oneway


# ------------------------------------------------------------------
# Significance helpers
# ------------------------------------------------------------------

def _stars(p: float) -> str:
    """Return significance stars for *p*-value."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def _p_fmt(p: float) -> str:
    """Format *p*-value with trailing stars."""
    if np.isnan(p):
        return "NA"
    stars = _stars(p)
    return f"{p:.4f}{stars}"


# ------------------------------------------------------------------
# Core ANOVA table builder
# ------------------------------------------------------------------

def anova_table(
    data: pd.DataFrame,
    cluster_labels: pd.Series,
    variables: Sequence[str],
    *,
    transform: Literal["none", "log", "boxcox"] = "none",
    label_col: str = "Variable",
) -> pd.DataFrame:
    """One-way ANOVA across clusters for every variable in *variables*.

    Parameters
    ----------
    data : pd.DataFrame
        Sites × variables matrix (reference sites only, same index as
        *cluster_labels*).
    cluster_labels : pd.Series
        1-indexed cluster assignment for each row in *data*.
    variables : list of str
        Which columns of *data* to analyse.
    transform : str
        Pre-ANOVA transformation applied to each variable independently:
        ``"none"`` (default), ``"log"`` (log₁₊x), ``"boxcox"``.
    label_col : str
        Name of the first column in the returned table (``"Variable"``
        or ``"Taxon"``).

    Returns
    -------
    pd.DataFrame
        Columns: *label_col*, ``SS Between``, ``df`` (between), ``SS Within``,
        ``df`` (within), ``F``, ``p``, ``Cluster C1``, ``Cluster C2``, …,
        followed by footer rows (blank, sample-size, blank, legend).
    """
    clusters = sorted(cluster_labels.unique())
    k = len(clusters)
    N = len(cluster_labels)
    df_between = k - 1
    df_within = N - k

    rows: list[dict] = []

    for var in variables:
        if var not in data.columns:
            continue
        col = data[var].copy()

        # --- optional pre-ANOVA transform --------------------------------
        if transform == "log":
            col = np.log1p(col.clip(lower=0))
        elif transform == "boxcox":
            from scipy.stats import boxcox as _boxcox
            shift = 0.0
            if col.min() <= 0:
                shift = abs(col.min()) + 1e-6
            try:
                vals, _ = _boxcox(col + shift)
                col = pd.Series(vals, index=col.index)
            except Exception:
                col = np.log1p(col.clip(lower=0))  # fallback

        # --- group stats -------------------------------------------------
        grand_mean = col.mean()
        groups = []
        for c in clusters:
            mask = cluster_labels == c
            groups.append(col.loc[mask].dropna())

        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)

        # scipy F-test
        if all(len(g) > 1 for g in groups):
            f_stat, p_val = f_oneway(*[g.values for g in groups])
        else:
            f_stat, p_val = np.nan, np.nan

        row = {
            label_col: var,
            "SS Between": round(ss_between, 2),
            "df": df_between,
            "SS Within": round(ss_within, 2),
            "df ": df_within,            # trailing space to distinguish
            "F": round(f_stat, 2) if not np.isnan(f_stat) else np.nan,
            "p": _p_fmt(p_val),
        }

        # per-cluster mean ± SEM (computed on the *untransformed* data so
        # the display values are interpretable)
        raw_col = data[var]
        for c in clusters:
            mask = cluster_labels == c
            vals = raw_col.loc[mask].dropna()
            m = vals.mean()
            se = vals.sem() if len(vals) > 1 else 0.0
            row[f"Cluster C{c}"] = f"{m:.2f} ± {se:.2f}"

        rows.append(row)

    df = pd.DataFrame(rows)

    # --- footer rows (blank, sample size, blank, legend) -----------------
    empty_row = {c: "" for c in df.columns}

    sample_row = {c: "" for c in df.columns}
    sample_row[label_col] = "Sample size (n)"
    for c in clusters:
        sample_row[f"Cluster C{c}"] = int((cluster_labels == c).sum())

    legend_row = {c: "" for c in df.columns}
    legend_row[label_col] = "Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1"

    footer = pd.DataFrame([empty_row, sample_row, empty_row, legend_row])
    df = pd.concat([df, footer], ignore_index=True)

    return df


# ------------------------------------------------------------------
# Convenience: extract raw p-value dict from an anova_table result
# ------------------------------------------------------------------

def extract_pvalues(
    table: pd.DataFrame,
    label_col: str = "Variable",
) -> dict[str, float]:
    """Return ``{variable_name: float_p}`` from an ANOVA summary table.

    Parses the ``p`` column (which may contain trailing stars) back to
    a plain float, skipping footer rows.
    """
    out: dict[str, float] = {}
    for _, row in table.iterrows():
        name = row.get(label_col, "")
        p_str = str(row.get("p", ""))
        if not name or not p_str or p_str == "NA" or name.startswith("Significance"):
            continue
        try:
            numeric = "".join(c for c in p_str if c in "0123456789.eE-+")
            out[str(name)] = float(numeric) if numeric else np.nan
        except ValueError:
            continue
    return out
