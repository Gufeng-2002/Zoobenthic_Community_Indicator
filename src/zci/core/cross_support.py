"""Cross-support classification: taxa × environment strength diagnostics.

Revised site-classification rules, binary strength flags, TaxaEnv_Class
assignment, classifier training on EnvStrong_TaxaStrong sites, and
structured evaluation on all four 2×2 classes.

Public API
----------
assign_taxa_status_v2
    Revised taxa status (Core/Peripheral/Uncertain) with configurable thresholds.
assign_taxa_strength
    Binary taxa strength (Strong/Weak) from Taxa_Status.
assign_taxa_env_class
    Four-way TaxaEnv_Class from binary strength flags.
build_updated_combined_table
    Full site-level summary with revised statuses and TaxaEnv_Class.
build_class_count_table
    2×2 class-count table (Output B).
evaluate_classifier_per_class
    Per-class confusion matrix and accuracy for a classifier.
build_model_performance_matrix
    2×2 model-performance structure (Output C).
build_model_comparison_row
    Single-row summary for one classifier (feeds Output D).
build_combined_confusion_table
    Combine per-class confusion matrices into one with MultiIndex.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


# ------------------------------------------------------------------
# Step 1 -- Revised taxa status
# ------------------------------------------------------------------

def assign_taxa_status_v2(
    taxa_silhouette: float,
    taxa_margin: float,
    sil_threshold: float = 0.2,
    margin_threshold: float = 0.2,
) -> str:
    """Core / Peripheral / Uncertain using revised thresholds.

    Core   : sil >= 0.2 AND margin >= 0.2
    Periph : sil >= 0   AND margin >= 0  (but not Core)
    Uncert : otherwise
    """
    if taxa_silhouette >= sil_threshold and taxa_margin >= margin_threshold:
        return "Core"
    elif taxa_silhouette >= 0 and taxa_margin >= 0:
        return "Peripheral"
    else:
        return "Uncertain"


# ------------------------------------------------------------------
# Step 2 -- Binary taxa strength
# ------------------------------------------------------------------

def assign_taxa_strength(taxa_status: str) -> str:
    """Strong if Core, Weak otherwise."""
    return "Strong" if taxa_status == "Core" else "Weak"


# ------------------------------------------------------------------
# Step 3 -- Four-way TaxaEnv_Class
# ------------------------------------------------------------------

def assign_taxa_env_class(env_strength: str, taxa_strength: str) -> str:
    return f"Env{env_strength}_Taxa{taxa_strength}"


# ------------------------------------------------------------------
# Updated combined table builder
# ------------------------------------------------------------------

def build_updated_combined_table(
    taxa_robustness: pd.DataFrame,
    env_robustness: pd.DataFrame,
    *,
    taxa_sil_threshold: float = 0.2,
    taxa_margin_threshold: float = 0.2,
) -> pd.DataFrame:
    """Build the full site-level summary with revised rules.

    Parameters
    ----------
    taxa_robustness : pd.DataFrame
        Original taxa robustness table (columns: Original_Cluster,
        Branch_AU, Silhouette, Own_Coassign, Best_Alt_Coassign, Margin, Status).
    env_robustness : pd.DataFrame
        Environmental robustness table (columns: Env_Silhouette,
        Env_Own_Coassign, Env_BestAlt_Coassign, Env_Margin, Env_Status).

    Returns
    -------
    pd.DataFrame
        Output A columns.
    """
    # Rename taxa columns with Taxa_ prefix
    taxa_rename = {
        "Silhouette": "Taxa_Silhouette",
        "Own_Coassign": "Taxa_Own_Coassign",
        "Best_Alt_Coassign": "Taxa_BestAlt_Coassign",
        "Margin": "Taxa_Margin",
        "Status": "Taxa_Status",
    }
    taxa = taxa_robustness.rename(columns=taxa_rename)
    combined = taxa.join(env_robustness, how="inner")

    # Step 1: Recompute Taxa_Status with revised rule
    combined["Taxa_Status"] = combined.apply(
        lambda r: assign_taxa_status_v2(
            r["Taxa_Silhouette"], r["Taxa_Margin"],
            taxa_sil_threshold, taxa_margin_threshold,
        ),
        axis=1,
    )

    # Step 2: Binary Taxa_Strength
    combined["Taxa_Strength"] = combined["Taxa_Status"].apply(assign_taxa_strength)

    # Step 3: Four-way class (Env_Strength is already in env_robustness)
    combined["TaxaEnv_Class"] = combined.apply(
        lambda r: assign_taxa_env_class(r["Env_Strength"], r["Taxa_Strength"]),
        axis=1,
    )

    return combined


# ------------------------------------------------------------------
# Output B -- 2×2 class count table
# ------------------------------------------------------------------

def build_class_count_table(combined: pd.DataFrame) -> pd.DataFrame:
    """2×2 site-count table (rows = Env_Strength, cols = Taxa_Strength)."""
    return pd.crosstab(
        combined["Env_Strength"],
        combined["Taxa_Strength"],
        margins=True,
        margins_name="Total",
    ).reindex(
        index=["Strong", "Weak", "Total"],
        columns=["Strong", "Weak", "Total"],
        fill_value=0,
    )


# ------------------------------------------------------------------
# Classifier evaluation helpers
# ------------------------------------------------------------------

_FOUR_CLASSES = [
    "EnvStrong_TaxaStrong",
    "EnvStrong_TaxaWeak",
    "EnvWeak_TaxaStrong",
    "EnvWeak_TaxaWeak",
]


def evaluate_classifier_per_class(
    combined: pd.DataFrame,
    predictions: pd.Series,
    probabilities: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict]:
    """Evaluate predictions within each of the four TaxaEnv classes.

    Parameters
    ----------
    combined : pd.DataFrame
        Must include Original_Cluster, TaxaEnv_Class.
    predictions : pd.Series
        Predicted cluster label per site (same index as combined).
    probabilities : pd.DataFrame or None
        Class probabilities per site (optional).

    Returns
    -------
    dict
        Keys = class names, values = dict with 'n_sites', 'accuracy',
        'confusion_matrix' (DataFrame), 'per_cluster_accuracy' (dict).
    """
    results: Dict[str, Dict] = {}
    all_clusters = sorted(combined["Original_Cluster"].unique())

    for cls in _FOUR_CLASSES:
        mask = combined["TaxaEnv_Class"] == cls
        subset = combined.loc[mask]
        n = len(subset)
        if n == 0:
            results[cls] = {
                "n_sites": 0,
                "accuracy": np.nan,
                "confusion_matrix": pd.DataFrame(),
                "per_cluster_accuracy": {},
            }
            continue

        y_true = subset["Original_Cluster"]
        y_pred = predictions.loc[subset.index]
        acc = accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=all_clusters)
        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{c}" for c in all_clusters],
            columns=[f"Pred_{c}" for c in all_clusters],
        )

        # Per-cluster accuracy within this class
        per_cluster: Dict[int, float] = {}
        for c in all_clusters:
            c_mask = y_true == c
            n_c = c_mask.sum()
            if n_c > 0:
                per_cluster[c] = float((y_pred.loc[c_mask] == c).sum() / n_c)

        results[cls] = {
            "n_sites": n,
            "accuracy": float(acc),
            "confusion_matrix": cm_df,
            "per_cluster_accuracy": per_cluster,
        }

    return results


# ------------------------------------------------------------------
# Output C -- 2×2 model-performance matrix (as DataFrame)
# ------------------------------------------------------------------

def build_model_performance_matrix(
    class_results: Dict[str, Dict],
) -> pd.DataFrame:
    """Flatten per-class evaluation into a summary DataFrame.

    Returns a table with one row per TaxaEnv_Class containing n_sites,
    accuracy, and per-cluster accuracy columns.
    """
    rows = []
    for cls in _FOUR_CLASSES:
        info = class_results.get(cls, {})
        row = {
            "TaxaEnv_Class": cls,
            "N_Sites": info.get("n_sites", 0),
            "Accuracy": info.get("accuracy", np.nan),
        }
        for c, pca in info.get("per_cluster_accuracy", {}).items():
            row[f"Cluster_{c}_PctCorrect"] = pca
        rows.append(row)
    return pd.DataFrame(rows).set_index("TaxaEnv_Class")


# ------------------------------------------------------------------
# Output D -- Model comparison row
# ------------------------------------------------------------------

def build_model_comparison_row(
    classifier_name: str,
    combined: pd.DataFrame,
    predictions: pd.Series,
    n_train: int,
) -> Dict:
    """One-row summary for a single classifier.

    Returns dict with keys matching Output D specification.
    """
    y_true = combined["Original_Cluster"]
    y_pred = predictions.loc[combined.index]
    env_strength = combined["Env_Strength"]

    overall_acc = accuracy_score(y_true, y_pred)

    # Env-weighted accuracy (Strong=1.0, Weak=0.5)
    weights = env_strength.map({"Strong": 1.0, "Weak": 0.5})
    correct = (y_true == y_pred).astype(float)
    weighted_acc = float((weights * correct).sum() / weights.sum())

    # Per-class accuracy
    per_class_acc = {}
    for cls in _FOUR_CLASSES:
        mask = combined["TaxaEnv_Class"] == cls
        if mask.sum() == 0:
            per_class_acc[cls] = np.nan
        else:
            per_class_acc[cls] = accuracy_score(
                y_true.loc[mask], y_pred.loc[mask]
            )

    return {
        "Classifier": classifier_name,
        "N_Train": n_train,
        "Overall_Accuracy": overall_acc,
        "Weighted_Accuracy": weighted_acc,
        "Acc_EnvStrong_TaxaStrong": per_class_acc.get("EnvStrong_TaxaStrong", np.nan),
        "Acc_EnvStrong_TaxaWeak": per_class_acc.get("EnvStrong_TaxaWeak", np.nan),
        "Acc_EnvWeak_TaxaStrong": per_class_acc.get("EnvWeak_TaxaStrong", np.nan),
        "Acc_EnvWeak_TaxaWeak": per_class_acc.get("EnvWeak_TaxaWeak", np.nan),
    }


# ------------------------------------------------------------------
# Combined confusion table (4 sub-matrices in one MultiIndex table)
# ------------------------------------------------------------------

def build_combined_confusion_table(
    class_results: Dict[str, Dict],
) -> pd.DataFrame:
    """Combine per-class confusion matrices into a single MultiIndex table.

    The outer row index spans Env strength (Strong / Weak), the outer
    column index spans Taxa strength (Strong / Weak).  Within each
    (Env, Taxa) cell sits the confusion matrix for that TaxaEnv class.

    Parameters
    ----------
    class_results : dict
        From :func:`evaluate_classifier_per_class`.

    Returns
    -------
    pd.DataFrame
        MultiIndex rows = (Env_Strength, True_label),
        MultiIndex cols = (Taxa_Strength, Pred_label).
    """
    env_levels = ["Strong", "Weak"]
    taxa_levels = ["Strong", "Weak"]

    pieces = []
    for env_lbl in env_levels:
        for taxa_lbl in taxa_levels:
            cls_key = f"Env{env_lbl}_Taxa{taxa_lbl}"
            info = class_results.get(cls_key, {})
            cm = info.get("confusion_matrix", pd.DataFrame())
            if cm.empty:
                continue
            # Add outer MultiIndex levels
            cm_mi = cm.copy()
            cm_mi.index = pd.MultiIndex.from_tuples(
                [(env_lbl, r) for r in cm.index],
                names=["Env_Strength", "True_Label"],
            )
            cm_mi.columns = pd.MultiIndex.from_tuples(
                [(taxa_lbl, c) for c in cm.columns],
                names=["Taxa_Strength", "Pred_Label"],
            )
            pieces.append(cm_mi)

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces)
