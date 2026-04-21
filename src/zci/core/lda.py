"""LDA computation — pure functions, no plotting, no file I/O.

Public API
----------
fit_lda
    Fit sklearn LDA on reference sites.
wilks_lambda_importance
    Per-variable Wilks' Lambda drop-one importance.
monte_carlo_cv
    Stratified repeated random-subsampling cross-validation.
predict_sites
    Predict cluster labels for new (non-reference) sites.
build_env_significance_table
    Publication-ready table: Habitat variable × Significance × Cluster means.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from ..models.lda import LDAFit, MCCVResult, WilksImportance


# ─── 1. Fit LDA ─────────────────────────────────────────────────────


def fit_lda(
    env_data: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    standardize: bool = True,
) -> LDAFit:
    """Fit sklearn LDA and return a :class:`LDAFit` container.

    Parameters
    ----------
    env_data : pd.DataFrame
        Environmental variables for reference sites (sites × vars).
    cluster_labels : array-like
        Cluster label per site (same length as *env_data*).
    standardize : bool
        Z-score env variables before fitting.

    Returns
    -------
    LDAFit
    """
    labels = np.asarray(cluster_labels)
    env = env_data.copy()

    scaler: Optional[StandardScaler] = None
    if standardize:
        scaler = StandardScaler()
        arr = scaler.fit_transform(env.values)
        env = pd.DataFrame(arr, index=env.index, columns=env.columns)

    lda = LinearDiscriminantAnalysis()
    lda.fit(env.values, labels)

    preds = lda.predict(env.values)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=sorted(np.unique(labels)))

    cluster_names = [f"Cluster {int(i)}" for i in sorted(np.unique(labels))]
    report = classification_report(
        labels, preds, target_names=cluster_names,
        output_dict=True, zero_division=0,
    )

    return LDAFit(
        model=lda,
        scaler=scaler,
        env_data=env,
        cluster_labels=labels,
        predictions=preds,
        accuracy=acc,
        confusion_matrix=cm,
        classification_report=report,
        explained_variance_ratio=lda.explained_variance_ratio_,
        env_variables=list(env.columns),
        cluster_names=cluster_names,
    )


# ─── 2. Wilks' Lambda variable importance ───────────────────────────


def _calc_wilks_lambda(X: np.ndarray, y: np.ndarray) -> float:
    """Wilks' Λ = |W| / |T|."""
    groups = np.unique(y)
    n_feat = X.shape[1]
    W = np.zeros((n_feat, n_feat))
    for g in groups:
        Xg = X[y == g]
        Xg_c = Xg - Xg.mean(axis=0)
        W += Xg_c.T @ Xg_c
    X_c = X - X.mean(axis=0)
    T = X_c.T @ X_c
    det_T = np.linalg.det(T)
    if det_T == 0:
        return 1.0
    return np.linalg.det(W) / det_T


def wilks_lambda_importance(lda_fit: LDAFit) -> WilksImportance:
    """Compute per-variable importance via Wilks' Lambda drop-one.

    For each variable, remove it from the model and measure how much
    Wilks' Λ increases (larger increase → more important variable).

    Returns
    -------
    WilksImportance
    """
    X = lda_fit.env_data.values
    y = lda_fit.cluster_labels
    env_vars = lda_fit.env_variables
    explained = lda_fit.explained_variance_ratio
    coefs = lda_fit.model.coef_          # (n_classes, n_features) or (n_classes-1, ...)

    n_samples = len(y)
    n_groups = len(np.unique(y))
    n_vars = len(env_vars)

    # ── axes summary ─────────────────────────────────────────────────
    cum = np.cumsum(explained)
    axes_summary = pd.DataFrame({
        "Axis": [f"LD{i+1}" for i in range(len(explained))],
        "Explained (%)": np.round(explained * 100, 2),
        "Cumulative (%)": np.round(cum * 100, 2),
    }).set_index("Axis")

    # ── drop-one importance ──────────────────────────────────────────
    wilks_full = _calc_wilks_lambda(X, y)

    rows: list[dict] = []
    sig_dict: Dict[str, Dict] = {}

    for i, var in enumerate(env_vars):
        mask = np.ones(n_vars, dtype=bool)
        mask[i] = False
        X_red = X[:, mask]

        wilks_red = _calc_wilks_lambda(X_red, y)
        delta = wilks_red - wilks_full

        df1 = n_groups - 1
        df2 = n_samples - n_groups - n_vars + 1
        f_stat = (delta / wilks_full) * (df2 / df1) if wilks_full > 0 and delta > 0 and df2 > 0 else 0.0
        p_val = 1 - stats.f.cdf(f_stat, df1, df2) if f_stat > 0 and df2 > 0 else 1.0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        sig_dict[var] = {"p_value": p_val, "significance": sig}

        row: dict = {
            "Environmental Variable": var,
            "Delta Wilks' Lambda": delta,
            "F-statistic": f_stat,
            "p-value": p_val,
            "Significance": sig,
        }
        # LD coefficients
        for k in range(coefs.shape[0]):
            row[f"LD{k+1} Coefficient"] = coefs[k, i]
        rows.append(row)

    imp_df = (
        pd.DataFrame(rows)
        .sort_values("F-statistic", ascending=False)
        .set_index("Environmental Variable")
    )

    # overall model significance (Bartlett)
    chi = -(n_samples - 1 - (n_vars + n_groups) / 2) * np.log(max(wilks_full, 1e-300))
    df_chi = n_vars * (n_groups - 1)
    overall_p = 1 - stats.chi2.cdf(chi, df_chi)

    return WilksImportance(
        axes_summary=axes_summary,
        variable_importance=imp_df,
        wilks_lambda_full=wilks_full,
        overall_significance={"chi_square": chi, "df": df_chi, "p_value": overall_p},
        significance_dict=sig_dict,
    )


# ─── 3. Monte Carlo Cross-Validation ────────────────────────────────


def monte_carlo_cv(
    env_data: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    standardize: bool = True,
    n_iterations: int = 1000,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> MCCVResult:
    """Stratified repeated random-subsampling CV.

    Each iteration: fit LDA on 80 % → predict on 20 % → store metrics.

    Returns
    -------
    MCCVResult
    """
    X = env_data.values.copy()
    y = np.asarray(cluster_labels)
    cluster_names = [f"Cluster {int(i)}" for i in sorted(np.unique(y))]

    if standardize:
        sc = StandardScaler()
        X = sc.fit_transform(X)

    sss = StratifiedShuffleSplit(n_splits=n_iterations, test_size=test_size,
                                 random_state=random_state)

    accuracies: list[float] = []
    all_preds: list = []
    all_true: list = []
    all_reports: list[dict] = []

    for train_idx, test_idx in sss.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_tr, y_tr)
        y_pred = lda.predict(X_te)

        accuracies.append(accuracy_score(y_te, y_pred))
        all_preds.extend(y_pred.tolist())
        all_true.extend(y_te.tolist())
        all_reports.append(classification_report(
            y_te, y_pred, target_names=cluster_names,
            labels=sorted(np.unique(y)),
            output_dict=True, zero_division=0,
        ))

    # aggregate confusion matrix over all folds
    agg_cm = confusion_matrix(all_true, all_preds, labels=sorted(np.unique(y)))

    # average classification report
    avg_report: Dict[str, Dict[str, float]] = {}
    for key in list(cluster_names) + ["weighted avg"]:
        metrics = {"precision": [], "recall": [], "f1-score": []}
        for rpt in all_reports:
            if key in rpt:
                for m in metrics:
                    metrics[m].append(rpt[key][m])
        avg_report[key] = {
            f"{m}_mean": float(np.mean(metrics[m])) if metrics[m] else 0.0
            for m in metrics
        }
        avg_report[key].update({
            f"{m}_std": float(np.std(metrics[m])) if metrics[m] else 0.0
            for m in metrics
        })

    return MCCVResult(
        mean_accuracy=float(np.mean(accuracies)),
        std_accuracy=float(np.std(accuracies)),
        median_accuracy=float(np.median(accuracies)),
        min_accuracy=float(np.min(accuracies)),
        max_accuracy=float(np.max(accuracies)),
        aggregate_confusion_matrix=agg_cm,
        avg_classification_report=avg_report,
        cluster_names=cluster_names,
        n_iterations=n_iterations,
        test_size=test_size,
        all_true_labels=all_true,
        all_predictions=all_preds,
    )


# ─── 4. Predict non-reference sites ─────────────────────────────────


def predict_sites(
    lda_fit: LDAFit,
    env_nonref: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Classify new sites using the fitted LDA.

    Uses the same scaler that was used during fitting.

    Returns
    -------
    predictions : pd.Series
        Predicted cluster label per site.
    probabilities : pd.DataFrame
        Class probabilities (sites × clusters).
    """
    X = env_nonref.copy()
    if lda_fit.scaler is not None:
        arr = lda_fit.scaler.transform(X)
        X = pd.DataFrame(arr, index=X.index, columns=X.columns)

    preds = lda_fit.model.predict(X.values)
    probs = lda_fit.model.predict_proba(X.values)

    pred_series = pd.Series(preds, index=env_nonref.index, name="Predicted_Cluster")
    prob_df = pd.DataFrame(
        probs, index=env_nonref.index,
        columns=lda_fit.cluster_names,
    )
    return pred_series, prob_df


# ─── 5. Publication-ready tables ─────────────────────────────────────


def build_env_significance_table(
    lda_fit: LDAFit,
    wilks: WilksImportance,
    raw_env: pd.DataFrame,
    cluster_labels: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Habitat variable | Significance level | Cluster C1 mean±SEM | …

    Uses *raw* (un-standardised) env values for the mean ± SEM columns,
    sorted by Wilks' Lambda p-value.

    Parameters
    ----------
    lda_fit : LDAFit
    wilks : WilksImportance
    raw_env : pd.DataFrame
        Environmental data in original scale (reference sites).
    cluster_labels : array-like
        Cluster label per reference site.
    """
    labels = np.asarray(cluster_labels)
    clusters = sorted(np.unique(labels))
    sig_dict = wilks.significance_dict
    env_vars = lda_fit.env_variables

    rows: list[dict] = []
    for var in env_vars:
        if var not in raw_env.columns:
            continue
        info = sig_dict.get(var, {"p_value": 1.0, "significance": "ns"})
        p = info["p_value"]

        if p < 0.001:
            sig_str = "p < 0.001***"
        elif p < 0.01:
            sig_str = "p < 0.01**"
        elif p < 0.05:
            sig_str = "p < 0.05*"
        else:
            sig_str = "p > 0.05"

        row: dict = {"Habitat variables": var, "Significance level": sig_str, "_p": p}
        for c in clusters:
            vals = raw_env.loc[labels == c, var].dropna() if isinstance(raw_env.index, pd.Index) else raw_env[var][labels == c]
            # reindex to match labels
            c_idx = np.where(labels == c)[0]
            vals = raw_env.iloc[c_idx][var].dropna()
            mean = vals.mean()
            sem = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            row[f"Cluster C{int(c)}"] = f"{mean:.2f} ± {sem:.2f}"
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("_p").drop(columns="_p").reset_index(drop=True)

    # footer rows
    sample_row = {col: "" for col in df.columns}
    sample_row["Habitat variables"] = "Sample size (n)"
    for c in clusters:
        sample_row[f"Cluster C{int(c)}"] = int((labels == c).sum())

    legend_row = {col: "" for col in df.columns}
    legend_row["Habitat variables"] = "Significance: *** p<0.001, ** p<0.01, * p<0.05"

    df = pd.concat([
        df,
        pd.DataFrame([{col: "" for col in df.columns}]),
        pd.DataFrame([sample_row]),
        pd.DataFrame([{col: "" for col in df.columns}]),
        pd.DataFrame([legend_row]),
    ], ignore_index=True)

    return df


def build_confusion_matrix_table(
    cm: np.ndarray,
    cluster_names: List[str],
    *,
    note: str = "",
) -> pd.DataFrame:
    """Formatted confusion matrix: Group | % Correct | Cluster C1 | … | Total."""
    n = len(cluster_names)
    row_totals = cm.sum(axis=1)
    diag = np.diag(cm)
    pct = np.where(row_totals > 0, (diag / row_totals) * 100, 0)

    rows: list[dict] = []
    for i, name in enumerate(cluster_names):
        cid = int(name.split()[-1])
        row: dict = {"Group": f"Cluster C{cid}", "% Correct": int(round(pct[i]))}
        for j, pn in enumerate(cluster_names):
            pid = int(pn.split()[-1])
            row[f"Cluster C{pid}"] = int(cm[i, j])
        rows.append(row)

    # total row
    total_correct = diag.sum()
    total_all = row_totals.sum()
    overall = int(round(total_correct / total_all * 100)) if total_all else 0
    col_totals = cm.sum(axis=0)
    total_row: dict = {"Group": "Total", "% Correct": overall}
    for j, pn in enumerate(cluster_names):
        pid = int(pn.split()[-1])
        total_row[f"Cluster C{pid}"] = int(col_totals[j])
    rows.append(total_row)

    df = pd.DataFrame(rows).set_index("Group")
    df.index.name = ""

    if note:
        empty = {c: "" for c in df.columns}
        note_row = empty.copy()
        note_row[df.columns[0]] = note
        footer = pd.DataFrame([empty, note_row], columns=df.columns)
        footer.index = ["", " "]
        df = pd.concat([df, footer])

    return df


# ─── Comparison confusion-matrix builders ────────────────────────────


def _model_site_desc(train_eval: pd.DataFrame, model_name: str,
                     is_weighted: bool = False) -> str:
    """Build e.g. 'Model A (fully trained on 35 core sites)' from training evaluation."""
    counts = train_eval["Status"].value_counts()
    n_core = int(counts.get("Core", 0))
    n_periph = int(counts.get("Peripheral", 0))
    n_uncertain = int(counts.get("Uncertain", 0))
    parts = []
    if n_core:
        parts.append(f"{n_core} core")
    if n_periph:
        parts.append(f"{n_periph} peripheral")
    if n_uncertain:
        parts.append(f"{n_uncertain} uncertain")
    desc = " + ".join(parts) + " sites"
    if is_weighted:
        desc += ", weighted"
    # Extract model letter from model_name (e.g. "Model A (Core Only)" -> "Model A")
    prefix = " ".join(model_name.split()[:2])
    return f"{prefix} (fully trained on {desc})"


def build_allref_comparison_table(
    sections: List[dict],
) -> pd.DataFrame:
    """Build all-ref-sites comparison confusion matrix table.

    Each element of *sections* is a dict with keys:
      header  : str           - section header
      cm      : np.ndarray    - confusion matrix
      cnames  : list[str]     - cluster names
      stats   : dict | None   - {n_heldout, accuracy, pmax_mean, deltap_mean}
    """
    col_clusters = [
        f"Cluster C{int(n.split()[-1])}" for n in sections[0]["cnames"]
    ]
    all_cols = ["% Correct"] + col_clusters

    parts: List[pd.DataFrame] = []
    for idx, sec in enumerate(sections):
        if idx > 0:
            blank = pd.DataFrame(
                [[""] * len(all_cols)], columns=all_cols, index=[""],
            )
            parts.append(blank)

        # Section header row
        header_data = [""] * len(all_cols)
        header_row = pd.DataFrame(
            [header_data], columns=all_cols, index=[sec["header"]],
        )
        parts.append(header_row)

        # CM body (no note)
        cm_df = build_confusion_matrix_table(sec["cm"], sec["cnames"], note="")
        parts.append(cm_df)

        # Held-out stats footer (always present)
        s = sec.get("stats")
        if s and s["n_heldout"] > 0:
            stats_data = {c: "" for c in all_cols}
            stats_data["% Correct"] = int(round(s["accuracy"] * 100))
            if len(col_clusters) >= 1:
                stats_data[col_clusters[0]] = f"p_max={s['pmax_mean']:.3f}"
            if len(col_clusters) >= 2:
                stats_data[col_clusters[1]] = f"Dp={s['deltap_mean']:.3f}"
            stats_idx = f"Held-out (n={s['n_heldout']})"
        else:
            stats_data = {c: "/" for c in all_cols}
            stats_data["% Correct"] = "/"
            stats_idx = "Held-out (n=0)"
        stats_row = pd.DataFrame(
            [list(stats_data.values())], columns=all_cols,
            index=[stats_idx],
        )
        parts.append(stats_row)

    return pd.concat(parts)


def build_cv_comparison_table(
    sections: List[dict],
) -> pd.DataFrame:
    """Build CV comparison confusion matrix table with median % correct.

    Each element of *sections* is a dict with keys:
      header    : str              - section header
      agg_cm    : np.ndarray       - aggregate confusion matrix
      fold_cms  : list[np.ndarray] - per-fold confusion matrices
      cnames    : list[str]        - cluster names
    """
    col_clusters = [
        f"Cluster C{int(n.split()[-1])}" for n in sections[0]["cnames"]
    ]
    all_cols = ["% Correct"] + col_clusters

    parts: List[pd.DataFrame] = []
    for idx, sec in enumerate(sections):
        if idx > 0:
            blank = pd.DataFrame(
                [[""] * len(all_cols)], columns=all_cols, index=[""],
            )
            parts.append(blank)

        # Section header row
        header_data = [""] * len(all_cols)
        header_row = pd.DataFrame(
            [header_data], columns=all_cols, index=[sec["header"]],
        )
        parts.append(header_row)

        # Compute median per-fold % correct per cluster
        agg_cm = sec["agg_cm"]
        fold_cms = sec["fold_cms"]
        cnames = sec["cnames"]

        if fold_cms is not None and len(fold_cms) > 0:
            fold_pcts: List[np.ndarray] = []
            fold_overall: List[float] = []
            for cm in fold_cms:
                rt = cm.sum(axis=1)
                diag = np.diag(cm)
                pct = np.where(rt > 0, (diag / rt) * 100, np.nan)
                fold_pcts.append(pct)
                total = cm.sum()
                if total > 0:
                    fold_overall.append(float(np.trace(cm)) / total * 100)
            median_pcts = np.nanmedian(np.array(fold_pcts), axis=0)
            median_overall = np.nanmedian(fold_overall)
        else:
            # Fall back to aggregate
            rt = agg_cm.sum(axis=1)
            diag = np.diag(agg_cm)
            median_pcts = np.where(rt > 0, (diag / rt) * 100, 0).astype(float)
            total = agg_cm.sum()
            median_overall = float(np.trace(agg_cm)) / total * 100 if total else 0

        # Build CM rows with median % correct
        rows: List[dict] = []
        col_totals = agg_cm.sum(axis=0)
        for i, name in enumerate(cnames):
            cid = int(name.split()[-1])
            row: dict = {
                "Group": f"Cluster C{cid}",
                "% Correct": int(round(median_pcts[i])),
            }
            for j, pn in enumerate(cnames):
                pid = int(pn.split()[-1])
                row[f"Cluster C{pid}"] = int(agg_cm[i, j])
            rows.append(row)

        total_row: dict = {
            "Group": "Total",
            "% Correct": int(round(median_overall)),
        }
        for j, pn in enumerate(cnames):
            pid = int(pn.split()[-1])
            total_row[f"Cluster C{pid}"] = int(col_totals[j])
        rows.append(total_row)

        cm_df = pd.DataFrame(rows).set_index("Group")
        cm_df.index.name = ""
        parts.append(cm_df)

    return pd.concat(parts)


def build_classification_report_table(
    lda_fit: LDAFit,
) -> pd.DataFrame:
    """Single-fit classification report table."""
    rpt = lda_fit.classification_report
    names = lda_fit.cluster_names
    acc = lda_fit.accuracy
    total_support = sum(rpt[n]["support"] for n in names if n in rpt)

    rows: list[dict] = []
    for n in names:
        if n in rpt:
            rows.append({
                "": n,
                "Precision": round(rpt[n]["precision"], 2),
                "Recall": round(rpt[n]["recall"], 2),
                "F1-Score": round(rpt[n]["f1-score"], 2),
                "Support": int(rpt[n]["support"]),
            })
    rows.append({"": "", "Precision": "", "Recall": "", "F1-Score": "", "Support": ""})
    rows.append({"": "Accuracy", "Precision": "–", "Recall": "–",
                 "F1-Score": round(acc, 2), "Support": total_support})
    for avg_key in ("macro avg", "weighted avg"):
        if avg_key in rpt:
            rows.append({
                "": avg_key.title(),
                "Precision": round(rpt[avg_key]["precision"], 2),
                "Recall": round(rpt[avg_key]["recall"], 2),
                "F1-Score": round(rpt[avg_key]["f1-score"], 2),
                "Support": total_support,
            })
    rows.append({"": "", "Precision": "", "Recall": "", "F1-Score": "", "Support": ""})
    rows.append({
        "": f"Note: Overall Accuracy = {acc:.4f}; Number of sites = {total_support}",
        "Precision": "", "Recall": "", "F1-Score": "", "Support": "",
    })
    return pd.DataFrame(rows).set_index("")


def build_mccv_classification_report_table(
    mccv: MCCVResult,
) -> pd.DataFrame:
    """MCCV classification report: mean ± std per class + accuracy stats."""
    rpt = mccv.avg_classification_report
    names = mccv.cluster_names
    n_iter = mccv.n_iterations

    # approximate support per iteration
    all_true = mccv.all_true_labels
    unique_c = sorted(set(all_true))
    counts = {c: sum(1 for l in all_true if l == c) / n_iter for c in unique_c}

    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.3f} ± {std:.3f}"

    rows: list[dict] = []
    for name in names:
        if name not in rpt:
            continue
        r = rpt[name]
        cid = int(name.split()[-1])
        rows.append({
            "": name,
            "Precision": _fmt(r["precision_mean"], r["precision_std"]),
            "Recall": _fmt(r["recall_mean"], r["recall_std"]),
            "F1-Score": _fmt(r["f1-score_mean"], r["f1-score_std"]),
            "Support": int(round(counts.get(cid, 0))),
        })
    rows.append({"": "", "Precision": "", "Recall": "", "F1-Score": "", "Support": ""})

    if "weighted avg" in rpt:
        wa = rpt["weighted avg"]
        total_support = sum(int(round(v)) for v in counts.values())
        rows.append({
            "": "Weighted Avg",
            "Precision": _fmt(wa["precision_mean"], wa["precision_std"]),
            "Recall": _fmt(wa["recall_mean"], wa["recall_std"]),
            "F1-Score": _fmt(wa["f1-score_mean"], wa["f1-score_std"]),
            "Support": total_support,
        })

    rows.append({"": "", "Precision": "", "Recall": "", "F1-Score": "", "Support": ""})
    rows.append({
        "": "Mean accuracy", "Precision": "", "Recall": "",
        "F1-Score": _fmt(mccv.mean_accuracy, mccv.std_accuracy), "Support": "",
    })
    rows.append({
        "": "Median accuracy", "Precision": "", "Recall": "",
        "F1-Score": f"{mccv.median_accuracy:.4f}", "Support": "",
    })
    rows.append({"": "", "Precision": "", "Recall": "", "F1-Score": "", "Support": ""})
    rows.append({
        "": f"Note: Metrics shown as mean ± std across {n_iter:,} CV iterations",
        "Precision": "", "Recall": "", "F1-Score": "", "Support": "",
    })
    return pd.DataFrame(rows).set_index("")
