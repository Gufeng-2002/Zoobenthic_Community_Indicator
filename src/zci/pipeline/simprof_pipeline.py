"""Stage 2 SIMPROF pipeline — Similarity Profile analysis.

Runs after the AU sweep and before Ward's final clustering.
Tests whether the reference-site assemblage and each cluster have
statistically significant internal structure.

Expected sequence in Stage 2
-----------------------------
Part 0 : pvclust AU sweep         (validates reference set size)
Part 0b: SIMPROF analysis         ← this module
Part 1 : Ward's clustering + robustness
Part 2 : Cross-support evaluation
Part 3 : Finalized LDA
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.clustering import select_reference_sites, ward_cluster
from ..core.transforms import (
    octave_transform,
    octave_to_chord,
    octave_to_hellinger,
    octave_to_log_chord,
    octave_to_relative_abundance,
)
from ..core.simprof import run_simprof_analysis, simprof_summary_table
from ..models.clustering import TAXA_COLUMNS
from ..viz.simprof_plots import (
    plot_simprof_profiles,
    plot_simprof_summary,
    plot_simprof_null_dist,
)


_TRANSFORMS = {
    "octave": octave_transform,
    "relative_abundance": octave_to_relative_abundance,
    "chord": octave_to_chord,
    "hellinger": octave_to_hellinger,
    "log_chord": octave_to_log_chord,
}


def simprof_pipeline(
    data_path: str | Path,
    stage1_artifact: str | Path,
    output_dir: str | Path,
    *,
    n_reference_sites: int = 52,
    taxa_transform: str = "octave",
    n_clusters_list: Sequence[int] = (2, 3),
    n_perm: int = 999,
    alpha: float = 0.05,
    metric: str = "braycurtis",
    file_prefix: str = "",
    save_plots: bool = True,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """Run SIMPROF analysis on reference sites, once per cluster count.

    For each k in *n_clusters_list*:
    1. Ward-cluster the reference sites into k groups.
    2. Run SIMPROF on the full reference set (should be significant →
       the assemblage is heterogeneous, confirming k > 1 clusters).
    3. Run SIMPROF on each of the k clusters (should NOT be significant
       → each cluster is internally homogeneous / terminal).
    4. Save summary table + figures.

    Parameters
    ----------
    data_path : path-like
        Path to the master Excel workbook.
    stage1_artifact : path-like
        Stage 1 output with pollution scores and site rankings.
    output_dir : path-like
        Base output directory (sub-folders are created automatically).
    n_reference_sites : int
        Number of least-polluted sites to include (must match AU-sweep
        and Ward's clustering settings).
    taxa_transform : str
        Taxa transformation; one of ``"octave"``, ``"chord"``,
        ``"hellinger"``, ``"log_chord"``, ``"relative_abundance"``.
    n_clusters_list : sequence of int
        Cluster counts to test (default ``(2, 3)``).
    n_perm : int
        Permutation replicates for each SIMPROF call (default 999).
    alpha : float
        Significance threshold (default 0.05).
    metric : str
        Distance metric: ``"braycurtis"`` (default) or ``"euclidean"``.
    file_prefix : str
        Optional prefix for output file names.
    save_plots : bool
        Whether to write figures to disk.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Keys are cluster counts (as strings, e.g. ``"k2"``); values are
        the list of per-group SIMPROF result dicts.
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    if verbose:
        print("  SIMPROF: loading data …")
    data_full = read_study_data(data_path)
    taxa_block = extract_block(data_full, "taxa", "raw")
    taxa_all = taxa_block[TAXA_COLUMNS].copy()

    stage1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
    score_cols = [
        c for c in stage1.columns
        if c[0] == "01_pollution_assessment" and c[1] == "raw"
        and c[2].endswith("_Score")
    ]
    if not score_cols:
        raise KeyError("No pollution score column found in Stage 1 artifact")
    pollution = stage1.loc[:, score_cols[0]]

    # Align taxa to sites present in the Stage 1 artifact (matches Ward pipeline)
    taxa_all = taxa_all.loc[taxa_all.index.intersection(pollution.index)]
    ref_mask = select_reference_sites(pollution, quantile=n_reference_sites)
    taxa_ref_raw = taxa_all.loc[ref_mask]

    # ── Transform taxa ───────────────────────────────────────────────
    transform_fn = _TRANSFORMS.get(taxa_transform)
    if transform_fn is None:
        raise ValueError(
            f"Unknown taxa_transform {taxa_transform!r}. "
            f"Choose from: {list(_TRANSFORMS)}"
        )
    taxa_ref_transformed = transform_fn(taxa_ref_raw)

    all_results: dict[str, list[dict]] = {}

    for k in n_clusters_list:
        prefix = f"{file_prefix}k{k}_"
        if verbose:
            print(f"\n{'='*60}")
            print(f"  SIMPROF  k={k}  (n_perm={n_perm}, metric={metric!r})")
            print(f"{'='*60}")

        # Ward cluster into k groups (temporary; SIMPROF uses these labels
        # to define within-cluster groups, not as final output)
        labels, _ = ward_cluster(taxa_ref_transformed, n_clusters=k)

        results = run_simprof_analysis(
            taxa_ref_transformed,
            labels,
            n_perm=n_perm,
            alpha=alpha,
            metric=metric,
            random_state=42,
            verbose=verbose,
        )
        all_results[f"k{k}"] = results

        # ── Summary table ────────────────────────────────────────────
        summary_df = simprof_summary_table(results)
        if verbose:
            print(f"\n  SIMPROF summary (k={k}):")
            print(summary_df.to_string(index=False))

        # save_table / save_figure expect a base path WITHOUT extension
        table_base = tables_dir / f"{prefix}simprof_summary"
        save_table(summary_df, table_base)
        if verbose:
            print(f"  Saved: {table_base}.xlsx")

        if not save_plots:
            continue

        # ── Profile plot ─────────────────────────────────────────────
        fig_profiles, _ = plot_simprof_profiles(
            results,
            title=f"SIMPROF Similarity Profiles  (k={k}, {metric})",
        )
        save_figure(fig_profiles, figures_dir / f"{prefix}simprof_profiles")
        if verbose:
            print(f"  Saved: {figures_dir / (prefix + 'simprof_profiles')}.png")

        # ── p-value summary bar chart ────────────────────────────────
        fig_summary, _ = plot_simprof_summary(
            results,
            alpha=alpha,
            title=f"SIMPROF p-value Summary  (k={k}, n_perm={n_perm})",
        )
        save_figure(fig_summary, figures_dir / f"{prefix}simprof_pvalues")
        if verbose:
            print(f"  Saved: {figures_dir / (prefix + 'simprof_pvalues')}.png")

        # ── Null-distribution histogram for the full reference set ───
        full_result = next(
            (r for r in results if r["group_label"] == "All reference sites"), None
        )
        if full_result is not None:
            fig_null, _ = plot_simprof_null_dist(full_result, figsize=(6, 4))
            save_figure(fig_null, figures_dir / f"{prefix}simprof_null_dist")
            if verbose:
                print(f"  Saved: {figures_dir / (prefix + 'simprof_null_dist')}.png")

    return all_results
