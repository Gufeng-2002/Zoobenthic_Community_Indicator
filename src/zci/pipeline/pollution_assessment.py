"""Stage 1 — Pollution Assessment Pipeline.

Orchestrates:  read → extract → transform → PCA → score → visualise → save.

This is the **only** module that touches both ``io`` and ``core``.

Now produces **two** competing site-level contamination scores:

* **SumRel** — sum of rescaled component scores
* **MaxRel** — maximum of rescaled component scores

Both are saved with their own prefix so downstream analyses (RDA) can
compare them on the same footing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import matplotlib.pyplot as plt

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.transforms import (
    log2_transform,
    score_sumrel,
    score_maxrel,
)
from ..core.pca import run_pca
from ..viz.pca_plots import plot_variance_explained, plot_ridge_loadings
from ..viz.map_plots import plot_corridor_bifurcation
from ..models.pca import PCAResult


# The 16 pollution variables from the 2008 study (default)
POLLUTION_VARS_2008: List[str] = [
    "Co", "Al", "Ni", "Mn", "Fe", "Cr",
    "Cu", "Hg", "Pb", "Zn", "total PCB",
    "Cd", "OCS", "p,p'-DDE", "As", "Ca",
]


@dataclass
class PollutionPipelineResult:
    """Container for Stage 1 outputs including both scoring rules."""
    pca_result: PCAResult
    sumrel_score: pd.Series
    maxrel_score: pd.Series
    data: pd.DataFrame          # original multi-index data


def pollution_pca_pipeline(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    pollution_vars: Sequence[str] = POLLUTION_VARS_2008,
    pollution_standardize: bool = True,
    n_components: int = 5,
    selected_pcs: Sequence[str] | None = None,
    composite_transform: str = "min-max",
    maps_dir: str | Path | None = None,
    threshold_quantile: int | float = 0.20,
    bifurcation_plot_func=None,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> PollutionPipelineResult:
    """Run the complete pollution-PCA stage and save outputs.

    Steps
    -----
    1. Read the 3-level MultiIndex Excel workbook.
    2. Extract the ``("chemical", "raw")`` block for the specified variables.
    3. Screen variables for negligible variation or extreme redundancy.
    4. Apply log₂(1 + x) transformation.
    5. Fit PCA, compute scaled loadings and standardised scores.
       PC signs are auto-oriented so higher = more contaminated.
    6. Compute **SumRel** and **MaxRel** contamination scores from
       the rescaled component scores.
    7. Save prefixed loadings, site-scores, and augmented data for each
       scoring rule to *output_dir/*.
    8. (Optional) Save variance-explained chart, ridge plot, and corridor
       bifurcation map for each scoring rule.

    Parameters
    ----------
    data_path : str or Path
        Path to the study-data Excel file.
    output_dir : str or Path
        Root output directory for this stage
        (e.g. ``results/01_pollution_assessment``).
    pollution_vars : sequence of str
        Which chemical columns to include.
    n_components : int
        Number of PCs to retain (default 5).
    selected_pcs : sequence of str or None
        Which PCs to include in the composite score.
        ``None`` → all *n_components* PCs.
    composite_transform : str
        Rescaling applied to the retained PCs before aggregation.
        ``"min-max"`` (default), ``"z-score"``, or ``"none"``.
    maps_dir : str, Path, or None
        Path to ``data/maps/`` folder with shapefiles.  ``None`` disables
        the corridor map figure.
    threshold_quantile : float
        Quantile (0–1) for the bifurcation cut.  Default ``0.20``.
    save_plots : bool
        Whether to produce and save figures.
    figure_formats : sequence of str
        Figure file formats (default ``("png",)``).
    table_formats : sequence of str
        Table file formats (default ``("xlsx",)``).
    verbose : bool
        Print progress messages.

    Returns
    -------
    PollutionPipelineResult
        Container with ``.pca_result``, ``.sumrel_score``, ``.maxrel_score``,
        and ``.data`` (original multi-index DataFrame).
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── 1. Read data ─────────────────────────────────────────────────────
    _log("[1/9] Reading study data …")
    data = read_study_data(data_path)
    _log(f"      {data.shape[0]} sites × {data.shape[1]} variables")

    # ── 2. Extract pollution block ────────────────────────────────────────
    _log("[2/9] Extracting pollution variables …")
    pollution_raw = extract_block(data, "chemical", "raw")[list(pollution_vars)]
    _log(f"      {pollution_raw.shape[1]} variables, {pollution_raw.shape[0]} sites")

    # ── 2b. Descriptive statistics of raw chemical concentrations ─────────
    _log("[2b/9] Computing descriptive statistics of raw chemical variables …")
    desc = pollution_raw.describe().T  # count, mean, std, min, 25%, 50%, 75%, max
    desc = desc.rename(columns={
        "25%": "Q1 (25%)",
        "50%": "Median (50%)",
        "75%": "Q3 (75%)",
    })
    desc["CV (%)"] = (desc["std"] / desc["mean"] * 100).round(2)
    desc["Range"] = desc["max"] - desc["min"]
    desc.index.name = "Chemical"
    # Reorder columns
    desc = desc[["count", "mean", "std", "CV (%)", "min",
                 "Q1 (25%)", "Median (50%)", "Q3 (75%)", "max", "Range"]]
    save_table(
        desc,
        tables_dir / "chemical_descriptive_stats",
        formats=table_formats,
        verbose=verbose,
    )

    # ── 3. Screen variables ───────────────────────────────────────────────
    _log("[3/9] Screening variables for negligible variation …")
    low_var = pollution_raw.std() < 1e-10
    if low_var.any():
        dropped = list(low_var[low_var].index)
        _log(f"      Dropping near-zero-variance columns: {dropped}")
        pollution_raw = pollution_raw.loc[:, ~low_var]
    else:
        _log("      All variables retained (no negligible-variance columns)")

    # ── 4. Transform ─────────────────────────────────────────────────────
    _log("[4/9] Applying log₂(1 + x) transformation …")
    pollution_transformed = log2_transform(pollution_raw)
    if pollution_standardize:
        _log("      Applying z-score standardisation to log-transformed variables …")
        pollution_transformed = (pollution_transformed - pollution_transformed.mean()) / pollution_transformed.std()

    # ── 5. PCA ────────────────────────────────────────────────────────────
    _log(f"[5/9] Fitting PCA (n_components={n_components}, orient_positive=True) …")
    result = run_pca(
        pollution_transformed,
        n_components=n_components,
        orient_positive=True,
    )

    cum_var = result.variance_info.loc["Cumulative Proportion"].iloc[-1]
    _log(f"      First {n_components} PCs explain "
         f"{float(cum_var) * 100:.1f} % of variance")

    # ── 6. Compute SumRel and MaxRel contamination scores ─────────────────
    if selected_pcs is None:
        selected_pcs = list(result.scores.columns)

    _log(f"[6/9] Computing SumRel & MaxRel scores "
         f"(PCs={list(selected_pcs)}, transform={composite_transform}) …")

    sumrel = score_sumrel(
        result.scores,
        selected_pcs=list(selected_pcs),
        transform=composite_transform,
    )
    maxrel = score_maxrel(
        result.scores,
        selected_pcs=list(selected_pcs),
        transform=composite_transform,
    )

    _log(f"      SumRel range: [{sumrel.min():.4f}, {sumrel.max():.4f}]")
    _log(f"      MaxRel range: [{maxrel.min():.4f}, {maxrel.max():.4f}]")

    # ── 7. Save tables + augmented artifacts (prefixed) ───────────────────
    _log("[7/9] Saving tables and artifacts …")

    # Common PCA outputs (shared between both scoring rules)
    save_table(
        result.loadings_with_variance(),
        tables_dir / "pc_loadings",
        formats=table_formats,
        verbose=verbose,
    )
    save_table(
        result.scores,
        tables_dir / "site_scores",
        formats=table_formats,
        verbose=verbose,
    )

    # Save per-score-rule outputs with prefix
    for prefix, score_series in [("SumRel", sumrel), ("MaxRel", maxrel)]:
        score_name = score_series.name  # e.g. "SumRel_Score"

        # Site rankings table
        ranking = score_series.sort_values().reset_index()
        ranking.columns = ["StationID", score_name]
        ranking["Rank"] = range(1, len(ranking) + 1)
        save_table(
            ranking,
            tables_dir / f"{prefix}_site_rankings",
            formats=table_formats,
            verbose=verbose,
        )

        # Build augmented DataFrame with MultiIndex columns
        augmented = result.to_augmented_dataframe(
            score_series,
            selected_pcs=list(selected_pcs),
            level0="01_pollution_assessment",
            level1="raw",
        )
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        augmented_path = artifacts_dir / f"{prefix}_01_updated_data.xlsx"
        augmented.to_excel(augmented_path)
        if verbose:
            print(f"  ✓ Saved augmented data: {augmented_path}")

    # ── 8. (Optional) Save PCA figures ────────────────────────────────────
    if save_plots:
        _log("[8/9] Saving PCA figures …")
        fig_var, _ = plot_variance_explained(result)
        save_figure(fig_var, figures_dir / "variance_explained",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_var)

        fig_ridge, _ = plot_ridge_loadings(result)
        save_figure(fig_ridge, figures_dir / "ridge_loadings",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_ridge)

    # ── 9. (Optional) Corridor maps for SumRel & MaxRel ──────────────────
    if save_plots and maps_dir is not None:
        _bifurc_func = bifurcation_plot_func or plot_corridor_bifurcation
        _log("[9/9] Saving corridor bifurcation maps (SumRel & MaxRel) …")
        sample_info = extract_block(data, "sample_info", "raw")

        for prefix, score_series in [("SumRel", sumrel), ("MaxRel", maxrel)]:
            fig_map, _ = _bifurc_func(
                scores=score_series,
                lat=sample_info["Latitude"],
                lon=sample_info["Longitude"],
                waterbody=sample_info["Waterbody"],
                maps_dir=maps_dir,
                threshold_quantile=threshold_quantile,
                score_label=f"{prefix} Contamination Score",
            )
            save_figure(fig_map, figures_dir / f"{prefix}_corridor_bifurcation",
                        formats=figure_formats, verbose=verbose)
            plt.close(fig_map)

    _log("\n✓ Pollution PCA pipeline complete (SumRel & MaxRel).")
    return PollutionPipelineResult(
        pca_result=result,
        sumrel_score=sumrel,
        maxrel_score=maxrel,
        data=data,
    )
