"""Stage 1 — Pollution Assessment Pipeline.

Orchestrates:  read → extract → transform → PCA → score → visualise → save.

This is the **only** module that touches both ``io`` and ``core``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd
import matplotlib.pyplot as plt

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.transforms import log2_transform, composite_pollution_score
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


def pollution_pca_pipeline(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    pollution_vars: Sequence[str] = POLLUTION_VARS_2008,
    n_components: int = 5,
    standardise_scores: str = "min-max",
    selected_pcs: Sequence[str] | None = None,
    composite_transform: str = "min-max",
    composite_weights: Dict[str, float] | Sequence[float] | None = None,
    maps_dir: str | Path | None = None,
    threshold_quantile: float = 0.20,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> PCAResult:
    """Run the complete pollution-PCA stage and save outputs.

    Steps
    -----
    1. Read the 3-level MultiIndex Excel workbook.
    2. Extract the ``("chemical", "raw")`` block for the specified variables.
    3. Apply log₂(1 + x) transformation.
    4. Fit PCA, compute scaled loadings and min-max-standardised scores.
    5. Compute composite pollution score from selected PCs.
    6. Save loadings table, site-scores table, and augmented data to
       *output_dir/*.
    7. (Optional) Save variance-explained chart and ridge plot to
       *output_dir/figures/*.

    Parameters
    ----------
    data_path : str or Path
        Path to the study-data Excel file.
    output_dir : str or Path
        Root output directory for this stage
        (e.g. ``results2/01_pollution_assessment``).
    pollution_vars : sequence of str
        Which chemical columns to include.
    n_components : int
        Number of PCs to retain (default 5).
    standardise_scores : str or None
        ``"min-max"`` (default), ``"z-score"``, or ``None``.
    selected_pcs : sequence of str or None
        Which PCs to include in the composite score.
        ``None`` → all *n_components* PCs.
    composite_transform : str
        Transformation applied to the selected PCs before summing.
        ``"min-max"`` (default) or ``"z-score"``.
    composite_weights : dict, list, or None
        Per-PC weights for the composite sum.  ``None`` → equal (all 1).
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
    PCAResult
        Structured result with ``.loadings``, ``.scores``,
        ``.scores_raw``, ``.variance_info``.
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── 1. Read data ─────────────────────────────────────────────────────
    _log("[1/8] Reading study data …")
    data = read_study_data(data_path)
    _log(f"      {data.shape[0]} sites × {data.shape[1]} variables")

    # ── 2. Extract pollution block ────────────────────────────────────────
    _log("[2/8] Extracting pollution variables …")
    pollution_raw = extract_block(data, "chemical", "raw")[list(pollution_vars)]
    _log(f"      {pollution_raw.shape[1]} variables, {pollution_raw.shape[0]} sites")

    # ── 3. Transform ─────────────────────────────────────────────────────
    _log("[3/8] Applying log₂(1 + x) transformation …")
    pollution_transformed = log2_transform(pollution_raw)

    # ── 4. PCA ────────────────────────────────────────────────────────────
    _log(f"[4/8] Fitting PCA (n_components={n_components}, "
         f"standardise={standardise_scores}) …")
    result = run_pca(
        pollution_transformed,
        n_components=n_components,
        standardise_scores=standardise_scores,
    )

    cum_var = result.variance_info.loc["Cumulative Proportion"].iloc[-1]
    _log(f"      First {n_components} PCs explain "
         f"{float(cum_var) * 100:.1f} % of variance")

    # ── 5. Composite pollution score ──────────────────────────────────────
    if selected_pcs is None:
        selected_pcs = list(result.scores.columns)
    _log(f"[5/8] Computing composite pollution score "
         f"(PCs={list(selected_pcs)}, transform={composite_transform}) …")
    pollution_score = composite_pollution_score(
        result.scores,
        selected_pcs=list(selected_pcs),
        transform=composite_transform,
        weights=composite_weights,
    )
    _log(f"      Score range: [{pollution_score.min():.4f}, {pollution_score.max():.4f}]")

    # ── 6. Save tables + augmented artifact ───────────────────────────────
    _log("[6/8] Saving tables and artifacts …")
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

    # Build augmented DataFrame with MultiIndex columns and save
    augmented = result.to_augmented_dataframe(
        pollution_score,
        selected_pcs=list(selected_pcs),
        level0="01_pollution_assessment",
        level1="raw",
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    augmented_path = artifacts_dir / "01_updated_data.xlsx"
    augmented.to_excel(augmented_path)
    if verbose:
        print(f"  ✓ Saved augmented data: {augmented_path}")

    # ── 7. (Optional) Save figures ────────────────────────────────────────
    if save_plots:
        _log("[7/8] Saving PCA figures …")
        fig_var, _ = plot_variance_explained(result)
        save_figure(fig_var, figures_dir / "variance_explained",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_var)

        fig_ridge, _ = plot_ridge_loadings(result)
        save_figure(fig_ridge, figures_dir / "ridge_loadings",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_ridge)

    # ── 8. (Optional) Corridor map + ECDF bifurcation figure ──────────────
    if save_plots and maps_dir is not None:
        _log("[8/8] Saving corridor bifurcation map …")
        # Extract lat / lon / waterbody from the original MultiIndex data
        sample_info = extract_block(data, "sample_info", "raw")
        fig_map, _ = plot_corridor_bifurcation(
            scores=pollution_score,
            lat=sample_info["Latitude"],
            lon=sample_info["Longitude"],
            waterbody=sample_info["Waterbody"],
            maps_dir=maps_dir,
            threshold_quantile=threshold_quantile,
            score_label="Pollution Score",
        )
        save_figure(fig_map, figures_dir / "corridor_bifurcation",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_map)

    _log("\n✓ Pollution PCA pipeline complete.")
    return result
