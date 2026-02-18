"""Redundancy Analysis (RDA) — pure linear-algebra computation, no plotting.

1. Multivariate OLS: Y ~ X  →  Ŷ = Xc @ B̂
2. PCA on Ŷ  →  constrained eigenvalues / eigenvectors
3. Three permutation-test methods: global, per-axis, per-term (Freedman–Lane)
4. Site / species / biplot score extraction (scaling 1 & 2)

Dataclasses (``RDAFit``, ``RDAScores``, ``PermutationTestResult``) live
in ``models.rda``; the :class:`RDA` class here is pure computation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..models.rda import RDAFit, RDAScores, PermutationTestResult

ArrayLike = Union[np.ndarray, pd.DataFrame]


# ─── RDA class ──────────────────────────────────────────────────────


class RDA:
    """Redundancy Analysis via linear algebra.

    Parameters
    ----------
    center_X, center_Y : bool
        Whether to mean-centre the predictor / response matrices.
    scale_X : bool
        Whether to z-score the predictor matrix before centering.
    ddof : int
        Degrees-of-freedom correction for covariance (default 1).
    """

    def __init__(
        self,
        *,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = False,
        ddof: int = 1,
    ) -> None:
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.ddof = ddof
        self.fit_: Optional[RDAFit] = None

    # ── fit ───────────────────────────────────────────────────────────

    def fit(self, X: ArrayLike, Y: ArrayLike) -> "RDA":
        """Fit RDA: multivariate OLS + PCA on fitted values.

        Parameters
        ----------
        X : array-like (n, p)
            Predictor (environmental) matrix.
        Y : array-like (n, q)
            Response (taxa / community) matrix.

        Returns
        -------
        self
        """
        X_df, Y_df = self._coerce_and_align(X, Y)

        if self.scale_X:
            X_df = (X_df - X_df.mean(axis=0)) / X_df.std(axis=0, ddof=1)

        Xc = self._center_df(X_df) if self.center_X else X_df.copy()
        Yc = self._center_df(Y_df) if self.center_Y else Y_df.copy()

        # OLS: Ŷ = Xc @ B̂
        B_hat = self._ols(Xc.to_numpy(), Yc.to_numpy())
        Y_hat = Xc.to_numpy() @ B_hat
        E = Yc.to_numpy() - Y_hat

        B_hat_df = pd.DataFrame(B_hat, index=Xc.columns, columns=Yc.columns)
        Y_hat_df = pd.DataFrame(Y_hat, index=Yc.index, columns=Yc.columns)
        E_df = pd.DataFrame(E, index=Yc.index, columns=Yc.columns)

        # PCA on Ŷ (covariance matrix)
        n = Y_hat_df.shape[0]
        cov_hat = (Y_hat_df.to_numpy().T @ Y_hat_df.to_numpy()) / (n - self.ddof)
        eigvals, eigvecs = np.linalg.eigh(cov_hat)

        tol = 1e-10
        keep = eigvals > tol
        eigvals = eigvals[keep]
        eigvecs = eigvecs[:, keep]

        if eigvals.size > 0:
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

        names = [f"RDA{i + 1}" for i in range(len(eigvals))]
        eigvals_s = pd.Series(eigvals, index=names)
        eigvecs_df = pd.DataFrame(eigvecs, index=Yc.columns, columns=names)

        total = eigvals_s.sum()
        explained = eigvals_s / total if total > 0 else eigvals_s * 0.0
        cumulative = explained.cumsum()

        inertia_total = float(np.sum(Yc.to_numpy() ** 2))
        inertia_con = float(np.sum(Y_hat ** 2))
        inertia_res = float(np.sum(E ** 2))
        r2 = inertia_con / inertia_total if inertia_total > 0 else float("nan")

        df_model = int(np.linalg.matrix_rank(Xc.to_numpy()))
        df_res = int(n - df_model - 1)
        r2_adj = (
            1.0 - (1.0 - r2) * ((n - 1) / df_res)
            if df_res > 0
            else float("nan")
        )

        self.fit_ = RDAFit(
            X=X_df, Y=Y_df,
            X_centered=Xc, Y_centered=Yc,
            coefficients=B_hat_df,
            Y_hat=Y_hat_df, residuals=E_df,
            constrained_eigenvalues=eigvals_s,
            constrained_eigenvectors=eigvecs_df,
            explained_proportion=explained,
            cumulative_explained=cumulative,
            inertia_total=inertia_total,
            inertia_constrained=inertia_con,
            inertia_residual=inertia_res,
            r2=r2, r2_adj=r2_adj,
            df_model=df_model, df_residual=df_res,
        )
        return self

    # ── scores ────────────────────────────────────────────────────────

    def scores(self, n_axes: Optional[int] = None) -> RDAScores:
        """Compute site, species, and biplot scores (scaling 1).

        Parameters
        ----------
        n_axes : int, optional
            How many RDA axes to retain (default: all positive).

        Returns
        -------
        RDAScores
        """
        fit = self._require_fit()
        eigvals = fit.constrained_eigenvalues
        eigvecs = fit.constrained_eigenvectors

        if eigvals.shape[0] == 0:
            raise RuntimeError("No positive constrained eigenvalues.")

        if n_axes is None:
            n_axes = int((eigvals > 0).sum())
        n_axes = max(1, min(n_axes, eigvals.shape[0]))

        lam = eigvals.iloc[:n_axes].to_numpy()
        A = eigvecs.iloc[:, :n_axes].to_numpy()

        inv_sqrt = np.diag(np.where(lam > 0, 1.0 / np.sqrt(lam), 0.0))
        sqrt_lam = np.diag(np.where(lam > 0, np.sqrt(lam), 0.0))

        U = fit.Y_hat.to_numpy() @ A @ inv_sqrt          # site scores
        V = A @ sqrt_lam                                  # species scores

        cols = eigvals.index[:n_axes]
        site_df = pd.DataFrame(U, index=fit.Y_hat.index, columns=cols)
        spec_df = pd.DataFrame(V, index=fit.Y_centered.columns, columns=cols)

        # Biplot scores: corr(Xc_std, U_std)
        Xc = fit.X_centered
        X_std = Xc / Xc.std(axis=0, ddof=1)
        U_std = site_df / site_df.std(axis=0, ddof=1)
        bp = (X_std.T @ U_std) / (Xc.shape[0] - 1)
        bp_df = pd.DataFrame(bp.to_numpy(), index=Xc.columns, columns=cols)

        return RDAScores(site_scores=site_df, species_scores=spec_df, biplot_scores=bp_df)

    def biplot_scores(self, n_axes: Optional[int] = None) -> pd.DataFrame:
        """Return biplot scores only (environmental variable loadings)."""
        return self.scores(n_axes).biplot_scores

    # ── permutation tests ─────────────────────────────────────────────

    def test_global(
        self,
        *,
        n_permutations: int = 999,
        random_state: Optional[int] = None,
    ) -> PermutationTestResult:
        """Global permutation test (pseudo-F, row permutation of Yc)."""
        fit = self._require_fit()
        obs_F = self._pseudo_f(
            fit.inertia_constrained, fit.df_model,
            fit.inertia_residual, fit.df_residual,
        )
        rng = np.random.default_rng(random_state)
        Xc = fit.X_centered.to_numpy()
        Yc = fit.Y_centered.to_numpy()
        null = np.empty(n_permutations)

        for i in range(n_permutations):
            Yp = Yc[rng.permutation(Yc.shape[0])]
            Bp = self._ols(Xc, Yp)
            Yhat_p = Xc @ Bp
            ss_c = float(np.sum(Yhat_p ** 2))
            ss_r = float(np.sum((Yp - Yhat_p) ** 2))
            null[i] = self._pseudo_f(ss_c, fit.df_model, ss_r, fit.df_residual)

        p = (1.0 + float(np.sum(null >= obs_F))) / (n_permutations + 1.0)
        return PermutationTestResult(obs_F, p, n_permutations, null)

    def test_axes(
        self,
        *,
        n_permutations: int = 999,
        random_state: Optional[int] = None,
        n_axes: Optional[int] = None,
    ) -> pd.DataFrame:
        """Per-axis permutation tests (max-reduction method).

        Returns DataFrame with columns: axis, eigenvalue, F, p.
        """
        fit = self._require_fit()
        eigvals = fit.constrained_eigenvalues
        n = fit.Y_centered.shape[0]

        if eigvals.shape[0] == 0:
            return pd.DataFrame({"axis": [], "eigenvalue": [], "F": [], "p": []})

        if n_axes is None:
            n_axes = int((eigvals > 0).sum())
        n_axes = max(1, min(n_axes, eigvals.shape[0]))

        axis_ss = (n - self.ddof) * eigvals.iloc[:n_axes].to_numpy()
        obs_F = (axis_ss / self.ddof) / (fit.inertia_residual / fit.df_residual)

        rng = np.random.default_rng(random_state)
        Xc = fit.X_centered.to_numpy()
        Yc = fit.Y_centered.to_numpy()
        null_F = np.empty((n_permutations, n_axes))
        tol = 1e-10

        for i in range(n_permutations):
            Yp = Yc[rng.permutation(Yc.shape[0])]
            Bp = self._ols(Xc, Yp)
            Yhat_p = Xc @ Bp
            Ep = Yp - Yhat_p
            ss_res_p = float(np.sum(Ep ** 2))

            cov_p = (Yhat_p.T @ Yhat_p) / (n - self.ddof)
            eig_p = np.linalg.eigvalsh(cov_p)
            eig_p = np.sort(eig_p[eig_p > tol])[::-1]

            if len(eig_p) < n_axes:
                eig_p = np.concatenate([eig_p, np.zeros(n_axes - len(eig_p))])
            else:
                eig_p = eig_p[:n_axes]

            null_F[i] = ((n - self.ddof) * eig_p) / (ss_res_p / fit.df_residual)

        # Max-reduction: compare each observed F with the max null F per perm
        null_max = null_F.max(axis=1, keepdims=True)
        null_max = np.broadcast_to(null_max, null_F.shape)
        pvals = (1.0 + (null_max >= obs_F).sum(axis=0)) / (n_permutations + 1.0)

        return pd.DataFrame({
            "axis": eigvals.index[:n_axes],
            "eigenvalue": eigvals.iloc[:n_axes].to_numpy(),
            "F": obs_F,
            "p": pvals,
        })

    def test_terms(
        self,
        *,
        n_permutations: int = 999,
        random_state: Optional[int] = None,
        terms: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Per-term Freedman–Lane permutation tests.

        Returns DataFrame with columns: term, delta_inertia, F, p.
        """
        fit = self._require_fit()
        Xc_full = fit.X_centered
        Yc = fit.Y_centered

        if terms is None:
            terms = list(Xc_full.columns)

        rng = np.random.default_rng(random_state)
        rows: List[Dict[str, Any]] = []

        for term in terms:
            if term not in Xc_full.columns:
                raise KeyError(f"Term '{term}' not in X columns")

            Xc_red = Xc_full.drop(columns=[term]).to_numpy()
            Xc_arr = Xc_full.to_numpy()
            Yc_arr = Yc.to_numpy()

            ss_con_red = self._fit_ss(Xc_red, Yc_arr)
            delta = fit.inertia_constrained - ss_con_red
            obs_F = self._pseudo_f(delta, 1, fit.inertia_residual, fit.df_residual)

            # Reduced-model residuals
            B_red = self._ols(Xc_red, Yc_arr)
            Yhat_red = Xc_red @ B_red
            E_red = Yc_arr - Yhat_red

            null = np.empty(n_permutations)
            for i in range(n_permutations):
                perm = rng.permutation(Yc_arr.shape[0])
                Yp = Yhat_red + E_red[perm]

                Bp_full = self._ols(Xc_arr, Yp)
                Yhat_p = Xc_arr @ Bp_full
                ss_c_full = float(np.sum(Yhat_p ** 2))
                ss_r_full = float(np.sum((Yp - Yhat_p) ** 2))

                ss_c_red = self._fit_ss(Xc_red, Yp)
                null[i] = self._pseudo_f(
                    ss_c_full - ss_c_red, 1, ss_r_full, fit.df_residual,
                )

            p = (1.0 + float(np.sum(null >= obs_F))) / (n_permutations + 1.0)
            rows.append({"term": term, "delta_inertia": delta, "F": obs_F, "p": p})

        return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)

    # ── variance partition ────────────────────────────────────────────

    def variance_partition(self) -> Dict[str, float]:
        """Return inertia / R² summary dict."""
        fit = self._require_fit()
        n = fit.Y_centered.shape[0]
        return {
            "inertia_total": fit.inertia_total,
            "inertia_constrained": fit.inertia_constrained,
            "inertia_residual": fit.inertia_residual,
            "r2": fit.r2,
            "r2_adj": fit.r2_adj,
            "trace_total_cov": fit.inertia_total / (n - self.ddof),
            "trace_constrained_cov": fit.inertia_constrained / (n - self.ddof),
            "trace_residual_cov": fit.inertia_residual / (n - self.ddof),
        }

    # ── private helpers ───────────────────────────────────────────────

    def _require_fit(self) -> RDAFit:
        if self.fit_ is None:
            raise RuntimeError("Call fit(X, Y) first.")
        return self.fit_

    @staticmethod
    def _center_df(df: pd.DataFrame) -> pd.DataFrame:
        return df - df.mean(axis=0)

    @staticmethod
    def _ols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return B

    @staticmethod
    def _pseudo_f(ss_num: float, df_num: int, ss_den: float, df_den: int) -> float:
        if df_num <= 0 or df_den <= 0 or ss_den <= 0:
            return float("nan")
        return (ss_num / df_num) / (ss_den / df_den)

    @staticmethod
    def _fit_ss(X: np.ndarray, Y: np.ndarray) -> float:
        """Constrained SS for a given X → Y."""
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        Yhat = X @ B
        return float(np.sum(Yhat ** 2))

    @staticmethod
    def _coerce_and_align(
        X: ArrayLike, Y: ArrayLike,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        Y_df = Y.copy() if isinstance(Y, pd.DataFrame) else pd.DataFrame(Y)

        if isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
            common = X_df.index.intersection(Y_df.index)
            if len(common) == 0:
                raise ValueError("No overlapping indices between X and Y.")
            X_df = X_df.loc[common]
            Y_df = Y_df.loc[common]

        if X_df.isna().any().any():
            raise ValueError("X contains NaN — impute/drop before RDA.")
        if Y_df.isna().any().any():
            raise ValueError("Y contains NaN — impute/drop before RDA.")

        return X_df, Y_df
