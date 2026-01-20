from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame]


@dataclass(frozen=True)
class PermutationTestResult:
    statistic: float
    p_value: float
    n_permutations: int
    null_distribution: np.ndarray


@dataclass(frozen=True)
class RDAScores:
    site_scores: pd.DataFrame
    species_scores: pd.DataFrame
    biplot_scores: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class RDAFit:
    X: pd.DataFrame
    Y: pd.DataFrame
    X_centered: pd.DataFrame
    Y_centered: pd.DataFrame
    coefficients: pd.DataFrame
    Y_hat: pd.DataFrame
    residuals: pd.DataFrame
    constrained_eigenvalues: pd.Series
    constrained_eigenvectors: pd.DataFrame
    explained_proportion: pd.Series
    cumulative_explained: pd.Series
    inertia_total: float
    inertia_constrained: float
    inertia_residual: float
    r2: float
    r2_adj: float
    df_model: int
    df_residual: int


class RDA:
    """Redundancy Analysis (RDA) via linear algebra.

    This class follows the same decomposition you implemented in the notebook:
    1) Fit multivariate OLS: Y ~ X
    2) Compute fitted community matrix Y_hat and residual matrix E
    3) PCA on Y_hat to obtain constrained eigenvalues/eigenvectors
    4) Compute site and species scores
    5) Variance partition and permutation tests

    Notes on scaling:
    - Inertia on *cross-product* scale: tr(Y^T Y) = ||Y||_F^2
    - Eigenvalues from a covariance matrix use a (n-1) denominator.
      For consistency: sum(eigenvalues) == ||Y_hat||_F^2/(n-1)
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

    def fit(self, X: ArrayLike, Y: ArrayLike) -> "RDA":
        X_df, Y_df = self._coerce_and_align(X, Y)

        if self.scale_X:
            X_df = (X_df - X_df.mean(axis=0)) / X_df.std(axis=0, ddof=1)

        Xc = self._center_df(X_df) if self.center_X else X_df.copy()
        Yc = self._center_df(Y_df) if self.center_Y else Y_df.copy()

        B_hat = self._ols_coefficients(Xc.to_numpy(), Yc.to_numpy())
        Y_hat = Xc.to_numpy() @ B_hat
        E = Yc.to_numpy() - Y_hat

        B_hat_df = pd.DataFrame(B_hat, index=Xc.columns, columns=Yc.columns)
        Y_hat_df = pd.DataFrame(Y_hat, index=Yc.index, columns=Yc.columns)
        E_df = pd.DataFrame(E, index=Yc.index, columns=Yc.columns)

        # PCA on the fitted matrix (constrained ordination)
        # cov_hat = (1/(n-1)) * Y_hat^T Y_hat
        n = Y_hat_df.shape[0]
        cov_hat = (Y_hat_df.to_numpy().T @ Y_hat_df.to_numpy()) / (n - self.ddof)
        eigvals, eigvecs = np.linalg.eigh(cov_hat)
        # Keep only strictly positive eigenpairs (numerical tolerance) and order descending.
        eigval_tol = 1e-10
        keep = eigvals > eigval_tol
        eigvals = eigvals[keep]
        eigvecs = eigvecs[:, keep]

        if eigvals.size > 0:
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

        eigvals_series = pd.Series(eigvals, index=[f"RDA{i+1}" for i in range(len(eigvals))])
        eigvecs_df = pd.DataFrame(eigvecs, index=Yc.columns, columns=eigvals_series.index)

        explained = eigvals_series / eigvals_series.sum() if eigvals_series.sum() > 0 else eigvals_series * 0.0
        cumulative = explained.cumsum()

        inertia_total = float(np.sum(Yc.to_numpy() ** 2))
        inertia_constrained = float(np.sum(Y_hat_df.to_numpy() ** 2))
        inertia_residual = float(np.sum(E_df.to_numpy() ** 2))

        r2 = inertia_constrained / inertia_total if inertia_total > 0 else float("nan")

        # Adjusted R^2 (Ezekiel-style; common in RDA reporting)
        # Uses df_model = number of predictors (rank) and an intercept.
        df_model = int(np.linalg.matrix_rank(Xc.to_numpy()))
        df_resid = int(n - df_model - 1)
        if df_resid <= 0:
            r2_adj = float("nan")
        else:
            r2_adj = 1.0 - (1.0 - r2) * ((n - 1) / df_resid)

        self.fit_ = RDAFit(
            X=X_df,
            Y=Y_df,
            X_centered=Xc,
            Y_centered=Yc,
            coefficients=B_hat_df,
            Y_hat=Y_hat_df,
            residuals=E_df,
            constrained_eigenvalues=eigvals_series,
            constrained_eigenvectors=eigvecs_df,
            explained_proportion=explained,
            cumulative_explained=cumulative,
            inertia_total=inertia_total,
            inertia_constrained=inertia_constrained,
            inertia_residual=inertia_residual,
            r2=r2,
            r2_adj=r2_adj,
            df_model=df_model,
            df_residual=df_resid,
        )
        return self

    def scores(self, n_axes: Optional[int] = None) -> RDAScores:
        fit = self._require_fit()

        eigvals = fit.constrained_eigenvalues
        eigvecs = fit.constrained_eigenvectors

        if eigvals.shape[0] == 0:
            raise RuntimeError(
                "No positive constrained eigenvalues were found; cannot compute RDA scores. "
                "This can happen if the fitted matrix Y_hat is (near-)zero."
            )

        if n_axes is None:
            n_axes = int((eigvals > 0).sum())
        n_axes = max(1, min(n_axes, eigvals.shape[0]))

        eigvals_k = eigvals.iloc[:n_axes].to_numpy()
        eigvecs_k = eigvecs.iloc[:, :n_axes].to_numpy()  # (p_taxa x k)

        # U = Y_hat A Lambda^{-1/2}
        # V = A Lambda^{1/2}
        # (avoid division by zero)
        inv_sqrt = np.diag([1.0 / np.sqrt(v) if v > 0 else 0.0 for v in eigvals_k])
        sqrt = np.diag([np.sqrt(v) if v > 0 else 0.0 for v in eigvals_k])

        U = fit.Y_hat.to_numpy() @ eigvecs_k @ inv_sqrt
        V = eigvecs_k @ sqrt

        site_scores = pd.DataFrame(U, index=fit.Y_hat.index, columns=eigvals.index[:n_axes])
        species_scores = pd.DataFrame(V, index=fit.Y_centered.columns, columns=eigvals.index[:n_axes])

        # Biplot scores (simple correlations between centered X and site scores)
        # (implementation-dependent; kept optional)
        Xc = fit.X_centered
        X_std = Xc / Xc.std(axis=0, ddof=1)
        U_std = site_scores / site_scores.std(axis=0, ddof=1)
        biplot = (X_std.T @ U_std) / (Xc.shape[0] - 1)
        biplot_scores = pd.DataFrame(biplot.to_numpy(), index=Xc.columns, columns=site_scores.columns)

        return RDAScores(site_scores=site_scores, species_scores=species_scores, biplot_scores=biplot_scores)

    def get_biplot_scores(self, n_axes: Optional[int] = None) -> pd.DataFrame:
        """
        Compute biplot scores (environmental variable loadings on RDA axes).
        
        These coefficients show how each habitat variable correlates with each 
        constrained ordination axis. They are used for biplot interpretation:
        - Larger absolute values indicate stronger association with an axis
        - Sign indicates direction of relationship
        
        This matches the output from R's vegan::scores(rda_model, display="bp")
        
        Parameters
        ----------
        n_axes : int, optional
            Number of RDA axes to compute scores for. If None, uses all positive axes.
            
        Returns
        -------
        pd.DataFrame
            Biplot scores with habitat variables as rows and RDA axes as columns.
        """
        fit = self._require_fit()
        
        eigvals = fit.constrained_eigenvalues
        
        if eigvals.shape[0] == 0:
            return pd.DataFrame()
        
        if n_axes is None:
            n_axes = int((eigvals > 0).sum())
        n_axes = max(1, min(n_axes, eigvals.shape[0]))
        
        # Compute site scores
        eigvals_k = eigvals.iloc[:n_axes].to_numpy()
        eigvecs_k = fit.constrained_eigenvectors.iloc[:, :n_axes].to_numpy()
        
        inv_sqrt = np.diag([1.0 / np.sqrt(v) if v > 0 else 0.0 for v in eigvals_k])
        U = fit.Y_hat.to_numpy() @ eigvecs_k @ inv_sqrt
        site_scores = pd.DataFrame(U, index=fit.Y_hat.index, columns=eigvals.index[:n_axes])
        
        # Biplot scores: correlations between environmental variables and site scores
        Xc = fit.X_centered
        X_std = Xc / Xc.std(axis=0, ddof=1)
        U_std = site_scores / site_scores.std(axis=0, ddof=1)
        biplot_coef = (X_std.T @ U_std) / (Xc.shape[0] - 1)
        
        return pd.DataFrame(biplot_coef.to_numpy(), index=Xc.columns, columns=site_scores.columns)

    def plot_biplot(
        self,
        axes: Tuple[int, int] = (1, 2),
        *,
        scaling: int = 1,
        show_sites: bool = True,
        show_species: bool = True,
        show_env: bool = True,
        species_labels: Optional[List[str]] = None,
        site_groups: Optional[pd.Series] = None,
        arrow_scale: float = 1.0,
        species_scale = 0.5,
        figsize: Tuple[float, float] = (10, 8),
        dpi = 100,
        **kwargs
    ):
        """
        Create an RDA biplot showing sites, species, and environmental vectors.
        
        Parameters
        ----------
        axes : tuple of int, default (1, 2)
            Which RDA axes to plot (1-indexed, e.g., (1, 2) for RDA1 vs RDA2).
        scaling : int, default 1
            Biplot scaling type (1 = distance biplot, 2 = correlation biplot).
        show_sites : bool, default True
            Whether to display site scores as points.
        show_species : bool, default True
            Whether to display species scores as text labels.
        show_env : bool, default True
            Whether to display environmental vectors as arrows.
        species_labels : list of str, optional
            Subset of species names to display. If None, shows all.
        site_groups : pd.Series, optional
            Categorical variable (e.g., waterbody) to color-code sites. 
            Index should match site_scores index.
        arrow_scale : float, default 1.0
            Scaling factor for environmental arrows (adjust for visibility).
        figsize : tuple, default (10, 8)
            Figure size (width, height) in inches.
        **kwargs
            Additional matplotlib arguments.
            
        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")
        
        fit = self._require_fit()
        
        # Convert 1-indexed to 0-indexed
        ax1_idx, ax2_idx = axes[0] - 1, axes[1] - 1
        axis_names = [fit.constrained_eigenvalues.index[ax1_idx], 
                      fit.constrained_eigenvalues.index[ax2_idx]]
        
        # Get scores (default scaling 1 from scores() method)
        scores_obj = self.scores(n_axes=max(axes))
        site_scores = scores_obj.site_scores.iloc[:, [ax1_idx, ax2_idx]].copy()
        species_scores = scores_obj.species_scores.iloc[:, [ax1_idx, ax2_idx]].copy()
        biplot_scores = scores_obj.biplot_scores.iloc[:, [ax1_idx, ax2_idx]].copy()
        
        # Apply scaling transformation
        # scores() returns scaling=1 by default: U = Y_hat A Λ^(-1/2), V = A Λ^(1/2)
        # For scaling=2: U = Y_hat A, V = A  (i.e., multiply U by Λ^(1/2), divide V by Λ^(1/2))
        if scaling == 2:
            eigvals = fit.constrained_eigenvalues.iloc[[ax1_idx, ax2_idx]].to_numpy()
            sqrt_lambda = np.sqrt(np.maximum(eigvals, 0))  # Avoid division by zero
            # Transform from scaling 1 to scaling 2
            site_scores = site_scores * sqrt_lambda  # Multiply by sqrt(λ)
            species_scores = species_scores / np.where(sqrt_lambda > 0, sqrt_lambda, 1.0)  # Divide by sqrt(λ)
        elif scaling != 1:
            raise ValueError(f"Scaling must be 1 or 2, got {scaling}")
        
        # Get explained variance for axis labels
        expl_var = fit.explained_proportion.iloc[[ax1_idx, ax2_idx]] * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
        if show_sites:
            if site_groups is not None:
                    # Align site_groups with site_scores
                    aligned_groups = site_groups.reindex(site_scores.index)
                    unique_groups = aligned_groups.dropna().unique()
                    unique_groups = sorted(unique_groups) # sort the unique cluster labels
                    
                    # Use a custom color order
                    import matplotlib.pyplot as plt
                    # First 5 colors are specified, then use colormap for remaining
                    base_colors = ['blue', 'red', 'green', 'orange', 'black']
                    if len(unique_groups) <= 5:
                        colors = base_colors[:len(unique_groups)]
                    else:
                        # Use tab10 colormap for additional groups beyond 5
                        extra_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups) - 5))
                        colors = base_colors + [extra_colors[i] for i in range(len(unique_groups) - 5)]
                    color_map = dict(zip(unique_groups, colors))
                    
                    # Plot each group separately for legend
                    for group in unique_groups:
                        mask = aligned_groups == group
                        group_sites = site_scores[mask]
                        ax.scatter(group_sites.iloc[:, 0], group_sites.iloc[:, 1],
                                c=[color_map[group]], s=60, alpha=0.6, 
                                edgecolors='gray', linewidth=0.5,
                                label= "Cluster " + str(group), zorder=3)
                        
                        # Add site labels with matching colors
                        for site in group_sites.index:
                            x = site_scores.loc[site, :].iloc[0] - 0.01
                            y = site_scores.loc[site, :].iloc[1] + 0.01
                            ax.text(x, y, site, fontsize=7, color='black', 
                                alpha=0.8, ha='center', va='bottom', zorder=4)
            else:
                # Default: all sites in black
                ax.scatter(site_scores.iloc[:, 0], site_scores.iloc[:, 1],
                            c='black', s=60, alpha=0.6, edgecolors='gray', linewidth=0.5,
                            label='Sites', zorder=3)
                # Add site labels
                for site in site_scores.index:
                    x = site_scores.loc[site, :].iloc[0] - 0.01
                    y = site_scores.loc[site, :].iloc[1] + 0.01
                    ax.text(x, y, site, fontsize=7, color='black', alpha=0.8)
            
        # Plot species scores
        if show_species:
            species_to_show = species_labels if species_labels is not None else species_scores.index
            for species in species_to_show:
                if species in species_scores.index:
                    x, y = species_scores.loc[species, :] * species_scale
                    ax.text(x, y, species, fontsize=10, color='red', alpha=0.7,
                           ha='center', va='center', zorder=2)
        
        # Plot environmental vectors
        if show_env:
            for var in biplot_scores.index:
                x, y = biplot_scores.loc[var, :] * arrow_scale
                
                # Get p-value for this variable if test_terms was run
                try:
                    term_results = self.test_terms(terms=[var], n_permutations=999)
                    p_val = term_results.loc[term_results['term'] == var, 'p'].values[0]
                    p_text = f" ($p={p_val:.3f}$)"
                    is_significant = p_val <= 0.05
                except:
                    p_text = ""
                    is_significant = True  # Default to significant style if p-value unavailable
                
                # Set style based on significance
                if is_significant:
                    linestyle = 'solid'
                    arrow_alpha = 0.7
                    text_alpha = 1.0
                else:
                    linestyle = 'dashed'
                    arrow_alpha = 0.5
                    text_alpha = 0.5
                
                # Draw arrow with appropriate style
                ax.arrow(0, 0, x, y, head_width=0.02, head_length=0.03,
                        fc='blue', ec='blue', alpha=arrow_alpha, linewidth=1.5, 
                        linestyle=linestyle, zorder=4)
                
                ax.text(x * 1.2, y * 1.2, var + p_text, fontsize=13, color='blue',
                       fontweight='bold', ha='center', va='center', alpha=text_alpha, zorder=5)
        
        # Add reference lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
        
        # Labels and styling
        ax.set_xlabel(f'{axis_names[0]} ({expl_var.iloc[0]:.1f}%)', fontsize=15, fontweight='bold')
        ax.set_ylabel(f'{axis_names[1]} ({expl_var.iloc[1]:.1f}%)', fontsize=15, fontweight='bold')
        # ax.set_title('RDA Biplot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)
        
        if show_sites:
            ax.legend(loc='best')
        
        plt.tight_layout()
        return fig, ax

    def variance_partition(self) -> Dict[str, float]:
        fit = self._require_fit()
        n = fit.Y_centered.shape[0]

        out: Dict[str, float] = {
            "inertia_total": fit.inertia_total,
            "inertia_constrained": fit.inertia_constrained,
            "inertia_residual": fit.inertia_residual,
            "r2": fit.r2,
            "r2_adj": fit.r2_adj,
            "trace_total_cov": fit.inertia_total / (n - self.ddof),
            "trace_constrained_cov": fit.inertia_constrained / (n - self.ddof),
            "trace_residual_cov": fit.inertia_residual / (n - self.ddof),
        }
        return out

    def test_global(
        self,
        *,
        n_permutations: int = 999,
        random_state: Optional[int] = None,
    ) -> PermutationTestResult:
        """Global permutation test for the overall RDA model.

        Statistic: pseudo-F = (SS_con/df_model) / (SS_res/df_res)
        Permutation: permute rows of Y (equivalent to Freedman–Lane under intercept-only reduced model).
        """
        fit = self._require_fit()

        obs_F = self._pseudo_f(fit.inertia_constrained, fit.df_model, fit.inertia_residual, fit.df_residual)

        rng = np.random.default_rng(random_state)
        null = np.empty(n_permutations, dtype=float)

        Xc = fit.X_centered.to_numpy()
        Yc = fit.Y_centered.to_numpy()

        for i in range(n_permutations):
            perm = rng.permutation(Yc.shape[0])
            Yp = Yc[perm, :]
            Bp = self._ols_coefficients(Xc, Yp)
            Yhat_p = Xc @ Bp
            Ep = Yp - Yhat_p
            ss_con = float(np.sum(Yhat_p ** 2))
            ss_res = float(np.sum(Ep ** 2))
            null[i] = self._pseudo_f(ss_con, fit.df_model, ss_res, fit.df_residual)

        p = (1.0 + float(np.sum(null >= obs_F))) / (n_permutations + 1.0)
        return PermutationTestResult(statistic=obs_F, p_value=p, n_permutations=n_permutations, null_distribution=null)

    def test_axes(
        self,
        *,
        n_permutations: int = 999,
        random_state: Optional[int] = None,
        n_axes: Optional[int] = None,
    ) -> pd.DataFrame:
        """Sequential permutation tests for constrained axes (RDA1, RDA2, ...).

        Statistic per axis k:
        pseudo-F_k = (SS_axis_k / 1) / (SS_res / df_res)

        Where SS_axis_k is the cross-product-scale inertia of axis k, i.e. (n-1)*lambda_k
        when lambda_k comes from a covariance matrix of Y_hat.
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
        obs_F = (axis_ss / self.ddof)  / (fit.inertia_residual / fit.df_residual)

        rng = np.random.default_rng(random_state)
        null_F = np.empty((n_permutations, n_axes), dtype=float)

        Xc = fit.X_centered.to_numpy()
        Yc = fit.Y_centered.to_numpy()
        eigval_tol = 1e-10

        for i in range(n_permutations):
            perm = rng.permutation(Yc.shape[0])
            Yp = Yc[perm, :]

            Bp = self._ols_coefficients(Xc, Yp)
            Yhat_p = Xc @ Bp
            Ep = Yp - Yhat_p
            ss_res_p = float(np.sum(Ep ** 2))

            cov_hat_p = (Yhat_p.T @ Yhat_p) / (n - self.ddof)
            eig_p_all = np.linalg.eigvalsh(cov_hat_p)
            # Apply same positive-eigenvalue filtering as fit()
            eig_p_positive = eig_p_all[eig_p_all > eigval_tol]
            eig_p_positive = np.sort(eig_p_positive)[::-1]
            
            # Pad with zeros if fewer positive eigenvalues than n_axes, or truncate
            if len(eig_p_positive) < n_axes:
                eig_p = np.concatenate([eig_p_positive, np.zeros(n_axes - len(eig_p_positive))])
            else:
                eig_p = eig_p_positive[:n_axes]
            
            axis_ss_p = (n - self.ddof) * eig_p
            null_F[i, :] = axis_ss_p / (ss_res_p / fit.df_residual)
        # replace all elements in each permutation with the largest randomized F-stat
        null_F_max_reduc = null_F.max(axis = 1, keepdims = True)
        null_F_max_reduc = np.broadcast_to(null_F_max_reduc, null_F.shape)    
        pvals = (1.0 + (null_F_max_reduc >= obs_F).sum(axis=0)) / (n_permutations + 1.0)
        out = pd.DataFrame(
            {
                "axis": eigvals.index[:n_axes],
                "eigenvalue": eigvals.iloc[:n_axes].to_numpy(),
                "F": obs_F,
                "p": pvals,
            }
        )
        return out

    def test_terms(
        self,
        *,
        n_permutations: int = 999,
        random_state: Optional[int] = None,
        terms: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Permutation tests for habitat variables (marginal term tests).

        Uses a Freedman–Lane style scheme:
        - Fit reduced model (drop one term)
        - Permute reduced-model residuals
        - Refit full and reduced models and compare constrained inertia

        Statistic:
        pseudo-F = (ΔSS_con / 1) / (SS_res_full / df_res)
        """
        fit = self._require_fit()

        Xc_full = fit.X_centered
        Yc = fit.Y_centered

        if terms is None:
            terms = list(Xc_full.columns)

        rng = np.random.default_rng(random_state)

        results: List[Dict[str, Any]] = []

        for term in terms:
            if term not in Xc_full.columns:
                raise KeyError(f"Term '{term}' not found in X columns")

            Xc_red = Xc_full.drop(columns=[term])
            df_model_full = fit.df_model
            df_res = fit.df_residual

            # Observed statistic
            ss_con_full = fit.inertia_constrained
            ss_res_full = fit.inertia_residual

            ss_con_red = self._fit_constrained_inertia(Xc_red.to_numpy(), Yc.to_numpy())
            delta_ss = ss_con_full - ss_con_red
            obs_F = self._pseudo_f(delta_ss, 1, ss_res_full, df_res)

            null = np.empty(n_permutations, dtype=float)

            # Reduced model residuals
            B_red = self._ols_coefficients(Xc_red.to_numpy(), Yc.to_numpy())
            Yhat_red = Xc_red.to_numpy() @ B_red
            E_red = Yc.to_numpy() - Yhat_red

            for i in range(n_permutations):
                perm = rng.permutation(Yc.shape[0])
                Yp = Yhat_red + E_red[perm, :]

                # Fit full model on permuted response
                Bp_full = self._ols_coefficients(Xc_full.to_numpy(), Yp)
                Yhat_p_full = Xc_full.to_numpy() @ Bp_full
                Ep_full = Yp - Yhat_p_full

                ss_con_p_full = float(np.sum(Yhat_p_full ** 2))
                ss_res_p_full = float(np.sum(Ep_full ** 2))

                # Fit reduced model on permuted response
                Bp_red = self._ols_coefficients(Xc_red.to_numpy(), Yp)
                Yhat_p_red = Xc_red.to_numpy() @ Bp_red
                ss_con_p_red = float(np.sum(Yhat_p_red ** 2))

                delta_p = ss_con_p_full - ss_con_p_red
                null[i] = self._pseudo_f(delta_p, 1, ss_res_p_full, df_res)

            p = (1.0 + float(np.sum(null >= obs_F))) / (n_permutations + 1.0)

            results.append(
                {
                    "term": term,
                    "delta_inertia": delta_ss,
                    "F": obs_F,
                    "p": p,
                }
            )

        return pd.DataFrame(results).sort_values("p")



    @staticmethod
    def _center_df(df: pd.DataFrame) -> pd.DataFrame:
        return df - df.mean(axis=0)

    @staticmethod
    def _ols_coefficients(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return B

    @staticmethod
    def _pseudo_f(ss_num: float, df_num: int, ss_den: float, df_den: int) -> float:
        if df_num <= 0 or df_den <= 0 or ss_den <= 0:
            return float("nan")
        return (ss_num / df_num) / (ss_den / df_den)

    @staticmethod
    def _fit_constrained_inertia(X: np.ndarray, Y: np.ndarray) -> float:
        B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        Yhat = X @ B
        return float(np.sum(Yhat ** 2))

    def _require_fit(self) -> RDAFit:
        if self.fit_ is None:
            raise RuntimeError("RDA has not been fit yet. Call fit(X, Y) first.")
        return self.fit_

    @staticmethod
    def _coerce_and_align(X: ArrayLike, Y: ArrayLike) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        Y_df = Y.copy() if isinstance(Y, pd.DataFrame) else pd.DataFrame(Y)

        # If both have indices, align them on intersection
        if isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
            common_idx = X_df.index.intersection(Y_df.index)
            if len(common_idx) == 0:
                raise ValueError("No overlapping indices between X and Y.")
            X_df = X_df.loc[common_idx]
            Y_df = Y_df.loc[common_idx]

        # Basic NA checks
        if X_df.isna().any().any():
            raise ValueError("X contains missing values; please impute/drop before RDA.")
        if Y_df.isna().any().any():
            raise ValueError("Y contains missing values; please impute/drop before RDA.")

        return X_df, Y_df
