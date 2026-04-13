"""Threshold grid search for cross-support classification.

Searches over sil_threshold, margin_threshold, env_sil_threshold,
env_margin_threshold to maximise LDA diagnostic accuracy on C1 + C3
(non-training reference sites).
"""

from __future__ import annotations

import itertools
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def _classify_sites(
    taxa_sil: pd.Series,
    taxa_margin: pd.Series,
    env_sil: pd.Series,
    env_margin: pd.Series,
    tsil_th: float,
    tmarg_th: float,
    esil_th: float,
    emarg_th: float,
) -> pd.Series:
    """Re-classify sites into 4 TaxaEnv classes given thresholds."""
    taxa_strong = (taxa_sil >= tsil_th) & (taxa_margin >= tmarg_th)
    env_strong = (env_sil > esil_th) & (env_margin > emarg_th)
    taxa_str = np.where(taxa_strong, "Strong", "Weak")
    env_str = np.where(env_strong, "Strong", "Weak")
    return pd.Series(
        ["Env" + e + "_Taxa" + t for e, t in zip(env_str, taxa_str)],
        index=taxa_sil.index,
    )


def _fit_lda_quick(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_all: pd.DataFrame,
) -> pd.Series:
    """Quick LDA fit + predict."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_a = scaler.transform(X_all)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_tr, y_train)
    return pd.Series(lda.predict(X_a), index=X_all.index)


def _eval_config(
    taxa_sil: pd.Series,
    taxa_margin: pd.Series,
    env_sil: pd.Series,
    env_margin: pd.Series,
    labels: pd.Series,
    env_raw: pd.DataFrame,
    tsil_th: float,
    tmarg_th: float,
    esil_th: float,
    emarg_th: float,
) -> dict | None:
    """Evaluate a single threshold configuration.

    Returns dict with accuracy metrics, or None if infeasible.
    """
    classes = _classify_sites(
        taxa_sil, taxa_margin, env_sil, env_margin,
        tsil_th, tmarg_th, esil_th, emarg_th,
    )
    train_mask = classes == "EnvStrong_TaxaStrong"
    n_train = int(train_mask.sum())
    if n_train < 3:
        return None

    train_sites = labels.index[train_mask]
    train_labels = labels.loc[train_sites]
    if train_labels.nunique() < 2:
        return None

    try:
        preds = _fit_lda_quick(env_raw.loc[train_sites], train_labels, env_raw)
    except Exception:
        return None

    # Diagnostic = non-training reference sites
    diag_mask = ~train_mask
    diag_sites = labels.index[diag_mask]

    diag_c1 = diag_sites[labels.loc[diag_sites] == 1]
    diag_c3 = diag_sites[labels.loc[diag_sites] == 3]

    c1_acc = (
        accuracy_score(labels.loc[diag_c1], preds.loc[diag_c1])
        if len(diag_c1) > 0 else 0.0
    )
    c3_acc = (
        accuracy_score(labels.loc[diag_c3], preds.loc[diag_c3])
        if len(diag_c3) > 0 else 0.0
    )

    return {
        "tsil": tsil_th,
        "tmarg": tmarg_th,
        "esil": esil_th,
        "emarg": emarg_th,
        "n_train": n_train,
        "LDA_diag_C1_acc": c1_acc,
        "LDA_diag_C3_acc": c3_acc,
        "LDA_C1C3_avg": (c1_acc + c3_acc) / 2,
        "LDA_overall_acc": accuracy_score(labels, preds),
    }


def grid_search_thresholds(
    combined: pd.DataFrame,
    env_strength_df: pd.DataFrame,
    labels: pd.Series,
    env_raw: pd.DataFrame,
    *,
    taxa_sil_vals: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    taxa_margin_vals: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    env_sil_vals: Sequence[float] = (-1.0, -0.5, -0.3, -0.1, 0.0),
    env_margin_vals: Sequence[float] = (-0.5, -0.3, -0.1, 0.0, 0.1, 0.2),
    verbose: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Search for threshold combination that maximises LDA C1+C3 diagnostic accuracy.

    Parameters
    ----------
    combined : DataFrame
        Ward's combined_robustness table (must have Taxa_Silhouette, Taxa_Margin).
    env_strength_df : DataFrame
        Environmental strength assignments (must have Env_Silhouette, Env_Margin).
    labels : Series
        Original cluster labels for reference sites.
    env_raw : DataFrame
        Raw (unstandardised) environmental variables for reference sites.

    Returns
    -------
    best : dict
        Best threshold configuration {tsil, tmarg, esil, emarg, n_train, ...}.
    results_df : DataFrame
        Full results table for all valid configurations.
    """
    taxa_sil = combined["Taxa_Silhouette"]
    taxa_margin = combined["Taxa_Margin"]
    env_sil = env_strength_df.loc[combined.index, "Env_Silhouette"]
    env_margin_col = env_strength_df.loc[combined.index, "Env_Margin"]

    grid = list(itertools.product(
        taxa_sil_vals, taxa_margin_vals, env_sil_vals, env_margin_vals,
    ))
    if verbose:
        print(f"  Grid search: {len(grid)} combinations ...")

    results = []
    for i, (ts, tm, es, em) in enumerate(grid):
        r = _eval_config(
            taxa_sil, taxa_margin, env_sil, env_margin_col,
            labels, env_raw, ts, tm, es, em,
        )
        if r is not None:
            results.append(r)

    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("Grid search found no valid configurations")

    best_idx = df["LDA_C1C3_avg"].idxmax()
    best = df.loc[best_idx].to_dict()

    if verbose:
        print(f"  {len(df)} valid configs out of {len(grid)}")
        print(f"  Best: tsil={best['tsil']}, tmarg={best['tmarg']}, "
              f"esil={best['esil']}, emarg={best['emarg']}")
        print(f"         n_train={int(best['n_train'])}, "
              f"LDA_C1={best['LDA_diag_C1_acc']:.1%}, "
              f"LDA_C3={best['LDA_diag_C3_acc']:.1%}, "
              f"avg={best['LDA_C1C3_avg']:.3f}")

    return best, df
