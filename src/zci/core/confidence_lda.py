"""Confidence-aware LDA — pure computation functions.

Public API
----------
compute_confidence_weights
    Soft, class-balanced confidence weight with min–max rescaling.
fit_weighted_lda
    Weighted multinomial logistic regression as LDA alternative.
evaluate_model_sites
    Per-site posterior diagnostics (p_max, delta_p).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..models.lda import LDAFit


# ─── 1. Confidence weights ──────────────────────────────────────────


def compute_confidence_weights(
    robustness: pd.DataFrame,
    *,
    alpha: float = 0.5,
    beta: float = 1.0,
    gamma: float = 0.7,
    delta: float = 0.7,
    lam: float = 0.5,
    epsilon: float = 0.15,
    normalize: bool = True,
) -> pd.Series:
    r"""Soft, class-balanced confidence weight.

    Rescaled silhouette and margin (min–max to [0, 1]):

    .. math::
        \tilde{s}_i = \mathrm{clip}\!\left(
            \frac{s_i - s_{\min}}{s_{\max} - s_{\min}},\; 0,\; 1\right)

        \tilde{M}_i = \mathrm{clip}\!\left(
            \frac{M_i - M_{\min}}{M_{\max} - M_{\min}},\; 0,\; 1\right)

    Confidence component:

    .. math::
        w_i^{(\mathrm{conf})}
        = \mathrm{AU}_i^{\alpha}\; A_i^{\beta}\;
          (\tilde{s}_i + \varepsilon)^{\gamma}\;
          (\tilde{M}_i + \varepsilon)^{\delta}

    Class-balanced weight:

    .. math::
        w_i = w_i^{(\mathrm{conf})} \times
              \left(\frac{\bar n}{n_{g(i)}}\right)^{\lambda}

    Parameters
    ----------
    robustness : pd.DataFrame
        Must contain columns: Branch_AU, Own_Coassign, Silhouette, Margin,
        Original_Cluster.
    alpha, beta, gamma, delta : float
        Exponents for AU, co-assign, rescaled silhouette, rescaled margin.
    lam : float
        Exponent for the class-balance multiplier.
    epsilon : float
        Positive floor added to rescaled silhouette and margin.
    normalize : bool
        If True, divide by mean weight for numerical stability.

    Returns
    -------
    pd.Series
        Weight per site, same index as *robustness*.
    """
    # --- Rescale silhouette and margin to [0, 1] via min–max ----------
    s = robustness["Silhouette"]
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        s_tilde = ((s - s_min) / (s_max - s_min)).clip(0, 1)
    else:
        s_tilde = pd.Series(0.5, index=s.index)

    m = robustness["Margin"]
    m_min, m_max = m.min(), m.max()
    if m_max > m_min:
        m_tilde = ((m - m_min) / (m_max - m_min)).clip(0, 1)
    else:
        m_tilde = pd.Series(0.5, index=m.index)

    # --- Confidence component -----------------------------------------
    au = robustness["Branch_AU"]
    a = robustness["Own_Coassign"]

    w_conf = (
        au.pow(alpha)
        * a.pow(beta)
        * (s_tilde + epsilon).pow(gamma)
        * (m_tilde + epsilon).pow(delta)
    )

    # --- Class-balance multiplier ------------------------------------
    clusters = robustness["Original_Cluster"]
    cluster_sizes = clusters.value_counts()
    n_bar = cluster_sizes.mean()
    balance = clusters.map(lambda g: (n_bar / cluster_sizes[g]) ** lam)
    w = w_conf * balance

    # --- Normalise ----------------------------------------------------
    if normalize:
        w_mean = w.mean()
        if w_mean > 0:
            w = w / w_mean

    w.name = "Weight"
    return w


# ─── 2. Fit LDA (standard — used for Model A and Model B) ───────────


def fit_lda_on_subset(
    env_data: pd.DataFrame,
    cluster_labels: pd.Series,
    *,
    standardize: bool = True,
) -> LDAFit:
    """Thin wrapper around sklearn LDA.  Same as core.lda.fit_lda but
    accepts a pd.Series for labels to keep index alignment explicit."""
    from ..core.lda import fit_lda
    return fit_lda(env_data, cluster_labels.values, standardize=standardize)


# ─── 3. Fit weighted classifier (Model C) ───────────────────────────


def fit_weighted_lda(
    env_data: pd.DataFrame,
    cluster_labels: pd.Series,
    sample_weights: pd.Series,
    *,
    standardize: bool = True,
) -> LDAFit:
    """Weighted multinomial logistic regression as a weighted LDA alternative.

    sklearn's ``LinearDiscriminantAnalysis`` does not accept sample weights,
    so we use ``LogisticRegression(multi_class='multinomial')`` instead and
    wrap the result in an :class:`LDAFit` for API compatibility.

    To keep the LDA-style LD-axis projection available we also fit an
    unweighted LDA on the same data and borrow its ``explained_variance_ratio_``
    and ``scalings_`` for downstream plotting.
    """
    labels = cluster_labels.values
    env = env_data.copy()

    scaler: StandardScaler | None = None
    if standardize:
        scaler = StandardScaler()
        arr = scaler.fit_transform(env.values)
        env = pd.DataFrame(arr, index=env.index, columns=env.columns)

    # Weighted multinomial logistic regression
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        C=1e6,           # very weak regularization to approximate LDA
        random_state=42,
    )
    w = sample_weights.loc[env.index].values
    lr.fit(env.values, labels, sample_weight=w)

    preds = lr.predict(env.values)
    acc = accuracy_score(labels, preds)
    unique_labels = sorted(np.unique(labels))
    cm = confusion_matrix(labels, preds, labels=unique_labels)
    cluster_names = [f"Cluster {int(i)}" for i in unique_labels]
    report = classification_report(
        labels, preds, target_names=cluster_names,
        output_dict=True, zero_division=0,
    )

    # Unweighted LDA for LD axes / scalings (used only for plotting)
    aux_lda = LinearDiscriminantAnalysis()
    aux_lda.fit(env.values, labels)
    evr = aux_lda.explained_variance_ratio_

    # Monkey-patch LR to look like an LDA for predict_sites/plotting:
    # we need .predict, .predict_proba (already there), .transform, .scalings_, .coef_
    lr.scalings_ = aux_lda.scalings_
    lr.explained_variance_ratio_ = evr
    lr.xbar_ = aux_lda.xbar_ if hasattr(aux_lda, "xbar_") else env.values.mean(axis=0)

    def _transform(X: np.ndarray) -> np.ndarray:
        return aux_lda.transform(X)
    lr.transform = _transform

    return LDAFit(
        model=lr,
        scaler=scaler,
        env_data=env,
        cluster_labels=labels,
        predictions=preds,
        accuracy=acc,
        confusion_matrix=cm,
        classification_report=report,
        explained_variance_ratio=evr,
        env_variables=list(env.columns),
        cluster_names=cluster_names,
    )


# ─── 4. Per-site posterior diagnostics ───────────────────────────────


def evaluate_sites(
    lda_fit: LDAFit,
    env_data: pd.DataFrame,
    true_labels: pd.Series,
    status: pd.Series,
    role: str,
) -> pd.DataFrame:
    """Predict sites and compute p_max and delta_p.

    Parameters
    ----------
    role : str
        'Train' or 'Uncertain' — stored in the output table.

    Returns
    -------
    pd.DataFrame with columns:
        Original_Cluster, Status, Predicted_Cluster,
        Prob_Cluster 1 .. Prob_Cluster k, p_max, delta_p, Role.
    """
    X = env_data.copy()
    if lda_fit.scaler is not None:
        arr = lda_fit.scaler.transform(X)
        X = pd.DataFrame(arr, index=X.index, columns=X.columns)

    preds = lda_fit.model.predict(X.values)
    probs = lda_fit.model.predict_proba(X.values)

    df = pd.DataFrame(index=env_data.index)
    df["Original_Cluster"] = true_labels.loc[env_data.index]
    df["Status"] = status.loc[env_data.index]
    df["Predicted_Cluster"] = preds

    for j, cname in enumerate(lda_fit.cluster_names):
        df[f"Prob_{cname}"] = probs[:, j]

    # p_max and delta_p
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    df["p_max"] = sorted_probs[:, 0]
    df["delta_p"] = sorted_probs[:, 0] - sorted_probs[:, 1]
    df["Role"] = role

    return df


# ─── 5. Cross-validated accuracy (within-subset) ────────────────────


def cross_validate_subset(
    env_data: pd.DataFrame,
    cluster_labels: pd.Series,
    *,
    sample_weights: pd.Series | None = None,
    standardize: bool = True,
    n_folds: int = 5,
    n_repeats: int = 10,
    random_state: int | None = 42,
    use_weighted_lr: bool = False,
) -> Tuple[float, float, np.ndarray, List[str], List[np.ndarray]]:
    """Repeated stratified k-fold CV.  Returns (mean_acc, std_acc, agg_cm, cluster_names, fold_cms)."""
    X = env_data.values.copy()
    y = cluster_labels.values
    unique_labels = sorted(np.unique(y))
    cluster_names = [f"Cluster {int(i)}" for i in unique_labels]

    accs: list[float] = []
    all_true: list = []
    all_pred: list = []
    fold_cms: list[np.ndarray] = []

    rng = np.random.RandomState(random_state)
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=rng.randint(0, 2**31))
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if standardize:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)

            if use_weighted_lr and sample_weights is not None:
                w_tr = sample_weights.iloc[train_idx].values
                model = LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000, C=1e6, random_state=42,
                )
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                model = LinearDiscriminantAnalysis()
                model.fit(X_tr, y_tr)

            y_pred = model.predict(X_te)
            accs.append(accuracy_score(y_te, y_pred))
            all_true.extend(y_te.tolist())
            all_pred.extend(y_pred.tolist())
            fold_cms.append(confusion_matrix(y_te, y_pred, labels=unique_labels))

    agg_cm = confusion_matrix(all_true, all_pred, labels=unique_labels)
    return float(np.mean(accs)), float(np.std(accs)), agg_cm, cluster_names, fold_cms
