"""Stage 1 — Pollution Assessment Pipeline.

Orchestrates:  read → extract → transform → PCA → visualise → save.

This is the **only** module that touches both ``io`` and ``core``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.transforms import log2_transform
from ..core.pca import run_pca
from ..viz.pca_plots import plot_variance_explained, plot_ridge_loadings
from ..models.results import PCAResult


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
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png"),
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
    5. Save loadings table and site-scores table to *output_dir/tables/*.
    6. (Optional) Save variance-explained chart and ridge plot to
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
    save_plots : bool
        Whether to produce and save figures.
    figure_formats : sequence of str
        Figure file formats (default ``("png", "pdf")``).
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

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── 1. Read data ─────────────────────────────────────────────────────
    _log("[1/5] Reading study data …")
    data = read_study_data(data_path)
    _log(f"      {data.shape[0]} sites × {data.shape[1]} variables")

    # ── 2. Extract pollution block ────────────────────────────────────────
    _log("[2/5] Extracting pollution variables …")
    pollution_raw = extract_block(data, "chemical", "raw")[list(pollution_vars)]
    _log(f"      {pollution_raw.shape[1]} variables, {pollution_raw.shape[0]} sites")

    # ── 3. Transform ─────────────────────────────────────────────────────
    _log("[3/5] Applying log₂(1 + x) transformation …")
    pollution_transformed = log2_transform(pollution_raw)

    # ── 4. PCA ────────────────────────────────────────────────────────────
    _log(f"[4/5] Fitting PCA (n_components={n_components}, "
         f"standardise={standardise_scores}) …")
    result = run_pca(
        pollution_transformed,
        n_components=n_components,
        standardise_scores=standardise_scores,
    )

    cum_var = result.variance_info.loc["Cumulative Proportion"].iloc[-1]
    _log(f"      First {n_components} PCs explain "
         f"{float(cum_var) * 100:.1f} % of variance")

    # ── 5. Save tables ────────────────────────────────────────────────────
    _log("[5/5] Saving tables …")
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

    # ── 6. (Optional) Save figures ────────────────────────────────────────
    if save_plots:
        _log("Saving figures …")
        fig_var, _ = plot_variance_explained(result)
        save_figure(fig_var, figures_dir / "variance_explained",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_var)

        fig_ridge, _ = plot_ridge_loadings(result)
        save_figure(fig_ridge, figures_dir / "ridge_loadings",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_ridge)

    _log("\n✓ Pollution PCA pipeline complete.")
    return result
