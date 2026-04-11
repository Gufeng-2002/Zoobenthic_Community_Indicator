"""Stage 1 — HZD (Hazard) Toxicity Scoring Pipeline.

Orchestrates:  read → extract chemical block → match benchmarks
→ compute McPhedran-style HZD effects → classify → save tables + figures.

Outputs (under ``HZD_Toxicity/``)
---------------------------------
tables/
    hzd_site_scores.xlsx          per-site HZD toxicity (%) + category
    hzd_chemical_effects.xlsx     per-site × per-chemical HZD effect matrix
    hzd_chemical_quotients.xlsx   per-site × per-chemical PEC quotient matrix
    hzd_benchmark_summary.xlsx    chemicals used + TEC / PEC + curve parameters
figures/
    HZD_corridor_bifurcation.png  spatial map coloured by HZD score
artifacts/
    HZD_01_updated_data.xlsx      Stage 1-style MultiIndex artifact with HZD score
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
    compute_hzd_curve_parameters,
    compute_hzd_effect_matrix,
    classify_hzd,
)


@dataclass
class HZDPipelineResult:
    """Container for HZD scoring outputs."""

    hzd_score: pd.Series
    """Per-site HZD toxicity score on the 0–100 scale."""

    hzd_category: pd.Series
    """Per-site hazard category on the HZD toxicity scale."""

    effect_matrix: pd.DataFrame
    """Sites × chemicals HZD effect percentages."""

    quotient_matrix: pd.DataFrame
    """Sites × chemicals diagnostic PEC quotient matrix."""

    benchmarks: pd.DataFrame
    """TEC / PEC lookup used."""

    data: pd.DataFrame
    """Original multi-index study data."""


def hzd_scoring_pipeline(
    data_path: str | Path,
    output_dir: str | Path,
    benchmark_path: str | Path,
    *,
    quotient_type: str | None = "TEC",
    maps_dir: str | Path | None = None,
    threshold_quantile: float = 0.20,
    bifurcation_plot_func=None,
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
    quotient_type : {"PEC", "TEC"} or None
        Retained only for backward compatibility. The McPhedran HZD method
        uses both TEC and PEC regardless of this argument.
    maps_dir : path or None
        Path to ``data/maps/`` folder for the corridor map.  ``None`` skips.
    threshold_quantile : float
        Quantile for the bifurcation map cut.
    """
    from ..viz.map_plots import plot_corridor_bifurcation

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"\n{'=' * 60}")
    _log("  HZD Toxicity Scoring (McPhedran hazard score)")
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
    _log("  [3] Extracting chemical data & computing HZD effects …")
    chem_all = extract_block(data, "chemical", "raw").apply(pd.to_numeric, errors="coerce")
    common = [c for c in benchmarks.index if c in chem_all.columns]
    _log(f"      Matched {len(common)} of {len(benchmarks)} benchmark chemicals: {common}")

    # ── 4. Compute HZD score ─────────────────────────────────────────
    if quotient_type not in (None, "PEC", "TEC"):
        raise ValueError(
            f"quotient_type must be None, 'PEC', or 'TEC', got {quotient_type!r}"
        )

    benchmark_summary = benchmarks.loc[common].copy()
    curve_params = compute_hzd_curve_parameters(benchmark_summary)
    effect_matrix = compute_hzd_effect_matrix(chem_all[common], benchmark_summary)
    hzd_score = effect_matrix.sum(axis=1, min_count=1).clip(lower=0.0, upper=100.0)
    hzd_score.name = "HZD_Toxicity_Percent"
    hzd_category = classify_hzd(hzd_score)

    benchmark_summary = benchmark_summary.join(curve_params)
    benchmark_summary["TEC_Effect_%"] = 5.0
    benchmark_summary["PEC_Effect_%"] = 50.0

    _log(
        f"      HZD toxicity (%) range: [{hzd_score.min():.4f}, {hzd_score.max():.4f}]"
    )
    _log(f"      Category counts:\n{hzd_category.value_counts().to_string()}")

    # Diagnostic PEC quotient matrix retained for comparability with prior output.
    quotient_matrix = chem_all[common].div(benchmark_summary["PEC"], axis=1)

    # ── 5. Save tables ───────────────────────────────────────────────
    _log("  [4] Saving tables …")

    # Site scores
    site_df = pd.DataFrame({
        "HZD_Toxicity_Percent": hzd_score,
        "category": hzd_category,
    })
    save_table(
        site_df, tables_dir / "hzd_site_scores",
        formats=table_formats, verbose=verbose,
    )

    save_table(
        effect_matrix, tables_dir / "hzd_chemical_effects",
        formats=table_formats, verbose=verbose,
    )

    save_table(
        quotient_matrix, tables_dir / "hzd_chemical_quotients",
        formats=table_formats, verbose=verbose,
    )

    save_table(
        benchmark_summary, tables_dir / "hzd_benchmark_summary",
        formats=table_formats, verbose=verbose,
    )

    # ── 6. Save Stage 1-style artifact ───────────────────────────────
    _log("  [5] Saving Stage 1 artifact …")
    hzd_score_named = hzd_score.rename("HZD_Score")
    hzd_category_named = hzd_category.rename("HZD_Category").astype(str)
    artifact_df = pd.DataFrame(
        {
            "HZD_Score": hzd_score_named,
            "HZD_Category": hzd_category_named,
        },
        index=hzd_score.index,
    )
    artifact_df.columns = pd.MultiIndex.from_tuples(
        [
            ("01_pollution_assessment", "raw", "HZD_Score"),
            ("01_pollution_assessment", "raw", "HZD_Category"),
        ],
        names=["block", "subblock", "var"],
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / "HZD_01_updated_data.xlsx"
    artifact_df.to_excel(artifact_path)
    if verbose:
        print(f"  ✓ Saved augmented data: {artifact_path}")

    # ── 7. Corridor map ──────────────────────────────────────────────
    if save_plots and maps_dir is not None:
        _log("  [6] Saving corridor map …")
        sample_info = extract_block(data, "sample_info", "raw")
        map_plotter = bifurcation_plot_func or plot_corridor_bifurcation

        fig_map, _ = map_plotter(
            scores=hzd_score,
            lat=sample_info["Latitude"],
            lon=sample_info["Longitude"],
            waterbody=sample_info["Waterbody"],
            maps_dir=maps_dir,
            threshold_quantile=threshold_quantile,
            score_label="HZD toxicity (%)",
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
        effect_matrix=effect_matrix,
        quotient_matrix=quotient_matrix,
        benchmarks=benchmark_summary,
        data=data,
    )
