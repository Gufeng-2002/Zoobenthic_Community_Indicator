"""Stage RDA — Redundancy Analysis pipeline.

Orchestrates:
  read original data → merge Stage 1 pollution scores → select reference
  sites → prepare env / taxa matrices → fit RDA → permutation tests
  → summary tables → triplot → save.

This pipeline is **independent** of the clustering work.  It can be
applied to any subset of sites with any target (taxa) and response
(environment) matrices.

Outputs
-------
tables/
    rda_axes_summary.xlsx   — eigenvalues, explained %, F, p per axis
    rda_terms_summary.xlsx  — delta inertia, F, p, biplot coefficients
figures/
    rda_triplot.png         — triplot (sites, taxa, env arrows)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.rda import RDA
from ..core.clustering import select_reference_sites, resolve_n_ref
from ..models.rda import RDAScores, PermutationTestResult, RDAResult
from ..viz.rda_plots import plot_rda_triplot


# ─── summary-table builders ────────────────────────────────────────


def _significance(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _build_axes_table(
    rda: RDA,
    axes_test: pd.DataFrame,
) -> pd.DataFrame:
    """Table 1 — Axis, Eigenvalue, Explained %, Cumulative %, F, p, Significance."""
    fit = rda.fit_
    n = len(axes_test)
    return pd.DataFrame({
        "Axis": [f"RDA{i + 1}" for i in range(n)],
        "Eigenvalue": fit.constrained_eigenvalues.iloc[:n].round(2).values,
        "Explained (%)": (fit.explained_proportion.iloc[:n] * 100).round(2).values,
        "Cumulative (%)": (fit.cumulative_explained.iloc[:n] * 100).round(2).values,
        "F-statistic": axes_test["F"].round(2).values,
        "p-value": axes_test["p"].round(2).values,
        "Significance": axes_test["p"].apply(_significance).values,
    })


def _build_terms_table(
    terms_test: pd.DataFrame,
    biplot_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Table 2 — Env Variable, Delta Inertia, F, p, Significance, RDA1/RDA2 coefs."""
    df = terms_test.copy()
    n_axes = min(2, biplot_scores.shape[1])
    for k in range(n_axes):
        col = biplot_scores.columns[k]
        df[f"{col} Coefficient"] = df["term"].map(
            lambda t, _c=col: round(biplot_scores.loc[t, _c], 2) if t in biplot_scores.index else np.nan
        )
    df["Significance"] = df["p"].apply(_significance)
    df = df.rename(columns={
        "term": "Environmental Variable",
        "delta_inertia": "Delta Inertia",
        "F": "F-statistic",
        "p": "p-value",
    })
    cols = [
        "Environmental Variable", "Delta Inertia", "F-statistic",
        "p-value", "Significance",
    ] + [c for c in df.columns if "Coefficient" in c]
    return df[cols].round(2)


# ─── main pipeline ──────────────────────────────────────────────────


def rda_pipeline(
    data_path: str | Path,
    stage1_artifact: str | Path | None = None,
    output_dir: str | Path = "results/RDA_analysis",
    *,
    pollution_score: pd.Series | None = None,
    score_column_name: str = "Pollution_Score",
    output_prefix: str = "",
    env_variables: Sequence[str] | None = None,
    taxa_columns: Sequence[str] | None = None,
    reference_quantile: int | float = 0.20,
    standardize_env: bool = True,
    log_transform_env: bool = False,
    taxa_transform: str = "octave",
    n_permutations: int = 999,
    random_state: int | None = 42,
    cluster_column: str | None = None,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> RDAResult:
    """Run the complete RDA pipeline and save outputs.

    The pipeline can receive the pollution score in two ways:

    * **Direct** — pass ``pollution_score`` (pd.Series) together with
      ``data_path``.  ``stage1_artifact`` can be ``None``.
    * **Legacy** — pass ``stage1_artifact`` path.  The score is read
      from the Excel artifact under the column ``score_column_name``.

    Parameters
    ----------
    data_path : path
        Original 3-level MultiIndex workbook.
    stage1_artifact : path or None
        ``01_updated_data.xlsx`` from Stage 1 (pollution scores).
        Not required when ``pollution_score`` is provided directly.
    output_dir : path
        Root for RDA outputs (``tables/``, ``figures/``).
    pollution_score : pd.Series, optional
        Pre-computed site-level contamination score (e.g. SumRel or
        MaxRel).  If provided, ``stage1_artifact`` is not read.
    score_column_name : str
        Column name to look for inside the Stage 1 artifact.  Only
        used when ``pollution_score is None``.
    output_prefix : str
        Prefix prepended to all output file names (e.g. ``"SumRel_"``).
    env_variables : list of str, optional
        Environmental column names.  ``None`` → sensible defaults.
    taxa_columns : list of str, optional
        Taxa column names.  ``None`` → all taxa in the data.
    reference_quantile : float
        Fraction of least-polluted sites to designate as reference.
    standardize_env : bool
        Z-score environmental variables.
    log_transform_env : bool
        ln(1 + x) before z-scoring.
    taxa_transform : str
        ``"octave"`` (identity) or ``"hellinger"``.
    n_permutations : int
        Number of permutations for all tests.
    random_state : int or None
        Seed for reproducibility.
    cluster_column : str or None
        If given, colours the triplot by cluster labels.
    save_plots / figure_formats / table_formats : misc
        Output control.
    verbose : bool
        Print progress.
    Returns
    -------
    RDAResult
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── 1. Read original data ────────────────────────────────────────
    _log("[1/10] Reading original study data …")
    data = read_study_data(data_path)
    _log(f"       {data.shape[0]} sites × {data.shape[1]} variables")

    # ── 2. Pollution scores → reference mask ─────────────────────────
    _log("[2/10] Obtaining pollution scores for reference selection …")
    if pollution_score is not None:
        _log("       Using directly-provided pollution score")
        ps = pollution_score.copy()
        ps.name = "Pollution_Score"
    elif stage1_artifact is not None:
        stage1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
        ps = stage1.loc[
            :, ("01_pollution_assessment", "raw", score_column_name)
        ]
        ps.name = "Pollution_Score"
    else:
        raise ValueError(
            "Either pollution_score or stage1_artifact must be provided"
        )

    ref_mask = select_reference_sites(ps, quantile=reference_quantile)
    n_ref = ref_mask.sum()
    _log(f"       {n_ref} reference sites (lowest {n_ref} sites)")

    # ── 3. Environmental matrix ──────────────────────────────────────
    if env_variables is None:
        env_variables = [
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ]
    _log(f"[3/10] Preparing {len(env_variables)} environmental variables …")

    env_all = extract_block(data, "environmental", "raw")
    env_vars_present = [v for v in env_variables if v in env_all.columns]
    env_ref = env_all.loc[ref_mask, env_vars_present].copy()

    if log_transform_env:
        _log("       Applying ln(1+x) to env variables")
        for col in env_ref.columns:
            mn = env_ref[col].min()
            shift = abs(mn) + 1e-6 if mn <= 0 else 0.0
            env_ref[col] = np.log(env_ref[col] + shift)

    env_scaler = None
    if standardize_env:
        _log("       Z-scoring environmental variables")
        env_scaler = StandardScaler()
        arr = env_scaler.fit_transform(env_ref)
        env_ref = pd.DataFrame(arr, index=env_ref.index, columns=env_ref.columns)

    # ── 4. Taxa matrix ───────────────────────────────────────────────
    _log("[4/10] Preparing taxa matrix …")
    taxa_all = extract_block(data, "taxa", "raw")
    if taxa_columns is not None:
        taxa_cols = [c for c in taxa_columns if c in taxa_all.columns]
        taxa_all = taxa_all[taxa_cols]
    taxa_ref = taxa_all.loc[ref_mask].copy()

    if taxa_transform == "hellinger":
        _log("       Hellinger transform")
        row_sums = taxa_ref.sum(axis=1)
        taxa_ref = taxa_ref.div(row_sums, axis=0).fillna(0).apply(np.sqrt)
    elif taxa_transform == "octave":
        _log("       Octave (identity) — data already in octave scale")
    elif taxa_transform == "none":
        _log("       No transformation applied")
    else:
        raise ValueError(f"Unknown taxa_transform={taxa_transform!r}")

    _log(f"       {taxa_ref.shape[0]} sites × {taxa_ref.shape[1]} taxa")

    # ── 5. Drop NaN rows ─────────────────────────────────────────────
    _log("[5/10] Dropping rows with NaN …")
    valid = env_ref.dropna().index.intersection(taxa_ref.dropna().index)
    env_ref = env_ref.loc[valid]
    taxa_ref = taxa_ref.loc[valid]
    _log(f"       {len(valid)} sites remaining")

    # ── 6. Fit RDA ───────────────────────────────────────────────────
    _log("[6/10] Fitting RDA (Y = taxa, X = env) …")
    rda = RDA(center_X=True, center_Y=True, scale_X=False, ddof=1)
    rda.fit(env_ref, taxa_ref)
    fit = rda.fit_
    _log(f"       R² = {fit.r2:.4f},  adj-R² = {fit.r2_adj:.4f}")
    _log(f"       Constrained inertia = {fit.inertia_constrained:.2f}")

    # ── 7. Permutation tests ─────────────────────────────────────────
    _log(f"[7/10] Permutation tests ({n_permutations} perms) …")
    global_test = rda.test_global(n_permutations=n_permutations, random_state=random_state)
    _log(f"       Global pseudo-F = {global_test.statistic:.2f}, "
         f"p = {global_test.p_value:.4f}")

    axes_test = rda.test_axes(n_permutations=n_permutations, random_state=random_state)
    terms_test = rda.test_terms(n_permutations=n_permutations, random_state=random_state)
    _log(f"       Significant axes (p<0.05): {(axes_test['p'] < 0.05).sum()}")
    _log(f"       Significant terms (p<0.05): {(terms_test['p'] < 0.05).sum()}")

    # ── 8. Summary tables ────────────────────────────────────────────
    _log("[8/10] Building summary tables …")
    rda_scores = rda.scores(n_axes=min(6, len(axes_test)))
    bp = rda.biplot_scores(n_axes=min(6, len(axes_test)))

    axes_table = _build_axes_table(rda, axes_test)
    terms_table = _build_terms_table(terms_test, bp)

    # ── 9. Triplot ───────────────────────────────────────────────────
    cluster_labels = None
    if cluster_column is not None:
        _log("[9/10] Loading cluster labels for triplot …")
        try:
            stage2_artifact_path = (
                Path(output_dir).parent / "02_taxa_assemblage" / "artifacts" / "02_hindsight_updated_data.xlsx"
            )
            stage2 = pd.read_excel(stage2_artifact_path, header=[0, 1, 2], index_col=0)
            cluster_labels = stage2.loc[
                valid, ("02_taxa_assemblage", "raw", cluster_column)
            ]
        except Exception as exc:
            _log(f"       ⚠ Could not load cluster labels: {exc}")

    if save_plots:
        _log("[9/10] Saving RDA triplot …")
        triplot_title = None
        if output_prefix:
            triplot_title = f"RDA Triplot — {output_prefix.rstrip('_')}"
        fig, _ = plot_rda_triplot(
            rda,
            axes=(1, 2),
            scaling=1,
            site_groups=cluster_labels,
            terms_test=terms_test,
            global_test=global_test,
            arrow_scale=2.0,
            species_scale=2.0,
            figsize=(12, 9),
            dpi=300,
            title=triplot_title,
        )
        save_figure(fig, figures_dir / f"{output_prefix}rda_triplot",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig)

    # ── 10. Save tables ──────────────────────────────────────────────
    _log("[10/10] Saving tables …")
    save_table(axes_table, tables_dir / f"{output_prefix}rda_axes_summary",
               formats=table_formats, verbose=verbose)
    save_table(terms_table, tables_dir / f"{output_prefix}rda_terms_summary",
               formats=table_formats, verbose=verbose)

    _log(f"\n✓ RDA pipeline complete.  "
         f"R²={fit.r2:.4f}, global p={global_test.p_value:.4f}")

    return RDAResult(
        rda_model=rda,
        scores=rda_scores,
        global_test=global_test,
        axes_test=axes_test,
        terms_test=terms_test,
        axes_table=axes_table,
        terms_table=terms_table,
        env_data=env_ref,
        taxa_data=taxa_ref,
        cluster_labels=cluster_labels,
        transformation_info={
            "standardize_env": standardize_env,
            "log_transform_env": log_transform_env,
            "taxa_transform": taxa_transform,
            "env_scaler": env_scaler,
            "n_sites": len(env_ref),
            "n_env": len(env_vars_present),
            "n_taxa": taxa_ref.shape[1],
        },
    )
