"""Stage 1 — HZD (Hazard) Toxicity Scoring Pipeline.

Orchestrates:  read → extract chemical block → match benchmarks
→ compute mean PEC quotient → classify → save tables + figures.

Outputs (under ``HZD_Toxicity/``)
---------------------------------
tables/
    hzd_site_scores.xlsx          per-site mean PEC-Q + category
    hzd_chemical_quotients.xlsx   per-site × per-chemical quotient matrix
    hzd_benchmark_summary.xlsx    chemicals used + TEC / PEC values
figures/
    HZD_corridor_bifurcation.png  spatial map coloured by HZD score
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import matplotlib.pyplot as plt

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.hzd import (
    load_tec_pec_benchmarks,
    compute_hzd_score,
    classify_hzd,
)


@dataclass
class HZDPipelineResult:
    """Container for HZD scoring outputs."""

    hzd_score: pd.Series
    """Per-site mean PEC quotient (higher = more hazardous)."""

    hzd_category: pd.Series
    """Per-site hazard category (Minimal / Low / Moderate / High)."""

    quotient_matrix: pd.DataFrame
    """Sites × chemicals quotient matrix."""

    benchmarks: pd.DataFrame
    """TEC / PEC lookup used."""

    data: pd.DataFrame
    """Original multi-index study data."""


def hzd_scoring_pipeline(
    data_path: str | Path,
    output_dir: str | Path,
    benchmark_path: str | Path,
    *,
    quotient_type: str = "PEC",
    maps_dir: str | Path | None = None,
    threshold_quantile: float = 0.20,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> HZDPipelineResult:
    """Compute HZD scores using consensus TEC/PEC benchmarks.

    Parameters
    ----------
    data_path : path
        Path to the 3-level MultiIndex Excel workbook.
    output_dir : path
        Root output directory (e.g. ``results/01_pollution_assessment/HZD_Toxicity``).
    benchmark_path : path
        Path to the TEC/PEC benchmark Excel file.
    quotient_type : {"PEC", "TEC"}
        Which benchmark for the quotient denominator.
    maps_dir : path or None
        Path to ``data/maps/`` folder for the corridor map.  ``None`` skips.
    threshold_quantile : float
        Quantile for the bifurcation map cut.
    """
    from ..viz.map_plots import plot_corridor_bifurcation

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"\n{'=' * 60}")
    _log("  HZD Toxicity Scoring (mean PEC quotient)")
    _log(f"{'=' * 60}")

    # ── 1. Read data ─────────────────────────────────────────────────
    _log("  [1] Reading study data …")
    data = read_study_data(data_path)
    _log(f"      {data.shape[0]} sites × {data.shape[1]} variables")

    # ── 2. Load benchmarks ───────────────────────────────────────────
    _log("  [2] Loading TEC/PEC benchmarks …")
    benchmarks = load_tec_pec_benchmarks(benchmark_path)
    _log(f"      {len(benchmarks)} chemicals in benchmark table")

    # ── 3. Extract chemical block and match ──────────────────────────
    _log("  [3] Extracting chemical data & computing quotients …")
    chem_all = extract_block(data, "chemical", "raw")
    common = [c for c in benchmarks.index if c in chem_all.columns]
    _log(f"      Matched {len(common)} of {len(benchmarks)} benchmark chemicals: {common}")

    # ── 4. Compute HZD score ─────────────────────────────────────────
    hzd_score = compute_hzd_score(
        chem_all, benchmarks, quotient_type=quotient_type,
    )
    hzd_category = classify_hzd(hzd_score)

    _log(f"      HZD (mean {quotient_type}-Q) range: "
         f"[{hzd_score.min():.4f}, {hzd_score.max():.4f}]")
    _log(f"      Category counts:\n{hzd_category.value_counts().to_string()}")

    # Quotient matrix for per-chemical detail
    bench_values = benchmarks.loc[common, quotient_type]
    quotient_matrix = chem_all[common].div(bench_values, axis=1)

    # ── 5. Save tables ───────────────────────────────────────────────
    _log("  [4] Saving tables …")

    # Site scores
    site_df = pd.DataFrame({
        f"mean_{quotient_type}_quotient": hzd_score,
        "category": hzd_category,
    })
    save_table(
        site_df, tables_dir / "hzd_site_scores",
        formats=table_formats, verbose=verbose,
    )

    # Chemical quotient matrix
    save_table(
        quotient_matrix, tables_dir / "hzd_chemical_quotients",
        formats=table_formats, verbose=verbose,
    )

    # Benchmark summary
    save_table(
        benchmarks.loc[common], tables_dir / "hzd_benchmark_summary",
        formats=table_formats, verbose=verbose,
    )

    # ── 6. Corridor map ──────────────────────────────────────────────
    if save_plots and maps_dir is not None:
        _log("  [5] Saving corridor map …")
        sample_info = extract_block(data, "sample_info", "raw")

        fig_map, _ = plot_corridor_bifurcation(
            scores=hzd_score,
            lat=sample_info["Latitude"],
            lon=sample_info["Longitude"],
            waterbody=sample_info["Waterbody"],
            maps_dir=maps_dir,
            threshold_quantile=threshold_quantile,
            score_label="HZD (mean PEC-Q)",
        )
        save_figure(
            fig_map, figures_dir / "HZD_corridor_bifurcation",
            formats=figure_formats, verbose=verbose,
        )
        plt.close(fig_map)

    _log("\n✓ HZD Toxicity scoring pipeline complete.")
    return HZDPipelineResult(
        hzd_score=hzd_score,
        hzd_category=hzd_category,
        quotient_matrix=quotient_matrix,
        benchmarks=benchmarks,
        data=data,
    )
