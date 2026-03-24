"""Stage 4 — Bray–Curtis NMDS + ZCI pipeline.

Orchestrates:
  read original data → merge Stage 1 pollution scores → merge Stage 3
  cluster labels → octave → relative abundance → build endpoints →
  iterative NMDS → PCA-rotate → species scores → ZCI construction →
  NMDS biplot → ZCI distribution figure → ZCI vs PS scatter →
  save tables, figures, artifacts.

Outputs
-------
tables/
    nmds_summary.xlsx
    zci_summary.xlsx
figures/
    nmds_biplot.png
    zci_distribution.png
    zci_vs_pollution.png
artifacts/
    04_updated_data.xlsx
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.transforms import octave_to_relative_abundance
from ..core.clustering import select_reference_sites
from ..core.nmds import run_nmds_cluster
from ..core.zci_scores import compute_zci, zci_correlation, ZCI_METHODS
from ..models.clustering import TAXA_COLUMNS
from ..models.nmds import ClusterNMDS, ClusterZCI, NMDSPipelineResult
from ..viz.nmds_plots import (
    plot_nmds_biplot,
    plot_zci_distribution,
    plot_zci_vs_pollution,
    _pollution_bin,
)


# ─── main pipeline ──────────────────────────────────────────────────


def nmds_pipeline(
    data_path: str | Path,
    stage1_artifact: str | Path,
    stage3_artifact: str | Path,
    output_dir: str | Path,
    *,
    # Endpoint sizes for the NMDS biplot (applied to ALL clusters equally)
    nmds_n_ep: int = 5,
    # Per-cluster ZCI configuration: {cluster_id: {"N_EP": int, "Method": str}}
    zci_config: Dict[int, Dict[str, Any]] | None = None,
    # NMDS parameters
    n_components: int = 2,
    n_nmds_iterations: int = 3,
    max_iter_per_run: int = 1000,
    n_init_first: int = 10,
    n_init_subsequent: int = 4,
    # Pollution-score percentile cut-offs
    reference_quantile: float = 0.20,
    # Misc
    random_state: int = 42,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> NMDSPipelineResult:
    """Run the complete Bray–Curtis NMDS + ZCI pipeline.

    Parameters
    ----------
    data_path : path
        Original 3-level MultiIndex workbook.
    stage1_artifact : path
        ``01_updated_data.xlsx`` (pollution scores).
    stage3_artifact : path
        ``03_updated_data.xlsx`` (cluster labels + Is_Reference).
    output_dir : path
        Root for outputs (``tables/``, ``figures/``, ``artifacts/``).
    nmds_n_ep : int
        Number of extreme sites per endpoint for the NMDS biplot.
    zci_config : dict, optional
        Per-cluster ZCI configuration.  Keys are cluster IDs (int),
        values are dicts with ``"N_EP"`` and ``"Method"`` keys.
        Available methods: ``"BC-Direct"``, ``"Distance"``,
        ``"Projection"``, ``"Centroid-Proj"``.

        Default::

            {1: {"N_EP": 15, "Method": "BC-Direct"},
             2: {"N_EP":  3, "Method": "BC-Direct"}}

    n_components : int
        NMDS dimensionality (default 2).
    n_nmds_iterations : int
        Number of iterative refinement passes (default 3).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    NMDSPipelineResult
    """
    output_dir = Path(output_dir)
    tables_dir  = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── Default ZCI config ───────────────────────────────────────────
    if zci_config is None:
        zci_config = {
            1: {"N_EP": 15, "Method": "BC-Direct"},
            2: {"N_EP":  3, "Method": "BC-Direct"},
        }

    # ── 1. Read original data ────────────────────────────────────────
    _log("[1/10] Reading original study data …")
    data = read_study_data(data_path)
    taxa_all = extract_block(data, "taxa", "raw")[list(TAXA_COLUMNS)]
    sample_info = extract_block(data, "sample_info", "raw")
    waterbody_all = sample_info["Waterbody"]
    waterbody_all.name = "Waterbody"
    _log(f"       taxa shape: {taxa_all.shape}")

    # ── 2. Pollution scores (Stage 1) ────────────────────────────────
    _log("[2/10] Reading Stage 1 artifact for pollution scores …")
    stage1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
    # Auto-detect score column (SumRel_Score, MaxRel_Score, or Pollution_Score)
    score_cols = [
        c for c in stage1.columns
        if c[0] == "01_pollution_assessment" and c[1] == "raw"
        and c[2].endswith("_Score")
    ]
    if not score_cols:
        raise KeyError("No pollution score column found in Stage 1 artifact")
    pollution_score = stage1.loc[:, score_cols[0]]
    pollution_score.name = "Pollution_Score"

    # ── 3. Cluster labels + reference flag (Stage 3) ─────────────────
    _log("[3/10] Reading Stage 3 artifact for cluster labels …")
    stage3 = pd.read_excel(stage3_artifact, header=[0, 1, 2], index_col=0)
    cluster_label = stage3.loc[
        :, ("03_lda_classification", "raw", "Predicted_Cluster")
    ]
    cluster_label.name = "Cluster"
    is_reference = stage3.loc[
        :, ("03_lda_classification", "raw", "Is_Reference")
    ]
    is_reference.name = "Is_Reference"

    # ── 4. Merge on common index ─────────────────────────────────────
    _log("[4/10] Merging data …")
    site_meta = pd.DataFrame({
        "Pollution_Score": pollution_score,
        "Cluster":         cluster_label,
        "Is_Reference":    is_reference,
        "Waterbody":       waterbody_all,
    })
    common_idx = taxa_all.index.intersection(site_meta.dropna().index)
    site_meta["Cluster"] = site_meta["Cluster"].astype("Int64")
    site_meta["Is_Reference"] = site_meta["Is_Reference"].astype("Int64")
    site_meta = site_meta.loc[common_idx]
    taxa = taxa_all.loc[common_idx]
    _log(f"       {len(common_idx)} sites retained")

    # ── 5. Octave → Relative Abundance ───────────────────────────────
    _log("[5/10] Converting octave → relative abundance …")
    taxa_relabd = octave_to_relative_abundance(taxa)

    clusters = sorted(site_meta["Cluster"].unique())

    # Pollution-level bins
    ps_all = site_meta["Pollution_Score"]
    p20 = np.percentile(ps_all, 20)
    p80 = np.percentile(ps_all, 80)
    site_meta["PS_Bin"] = site_meta["Pollution_Score"].apply(
        lambda x: _pollution_bin(x, p20, p80)
    )
    _log(f"       P20 = {p20:.3f},  P80 = {p80:.3f}")

    # ── 6. NMDS per cluster ──────────────────────────────────────────
    _log("[6/10] Running iterative NMDS per cluster …")
    nmds_dict: Dict[int, ClusterNMDS] = {}
    for cl in clusters:
        cl_sites = site_meta.index[site_meta["Cluster"] == cl]
        _log(f"       Cluster {cl}: {len(cl_sites)} sites, "
             f"N_EP={nmds_n_ep}, {n_nmds_iterations} NMDS iterations …")

        res = run_nmds_cluster(
            taxa_ra_cluster=taxa_relabd.loc[cl_sites],
            pollution_scores_cluster=site_meta.loc[cl_sites, "Pollution_Score"],
            taxa_columns=list(TAXA_COLUMNS),
            n_ep=nmds_n_ep,
            cluster_id=cl,
            n_components=n_components,
            n_iterations=n_nmds_iterations,
            max_iter_per_run=max_iter_per_run,
            n_init_first=n_init_first,
            n_init_subsequent=n_init_subsequent,
            random_state=random_state,
        )

        nmds_dict[cl] = ClusterNMDS(
            cluster_id=cl,
            coords_df=res["coords_df"],
            stress=res["stress"],
            var_explained=res["var_exp"],
            wa_df=res["wa_df"],
            ref_label=res["ref_label"],
            deg_label=res["deg_label"],
            ref_ids=res["ref_ids"],
            deg_ids=res["deg_ids"],
            n_ep=nmds_n_ep,
            flipped=res["flipped"],
        )
        _log(f"         stress = {res['stress']:.5f}"
             f"  {'(flipped)' if res['flipped'] else ''}")

    # ── 7. ZCI per cluster ───────────────────────────────────────────
    _log("[7/10] Computing ZCI per cluster …")
    zci_dict: Dict[int, ClusterZCI] = {}
    for cl in clusters:
        cfg = zci_config.get(cl, {"N_EP": 5, "Method": "Centroid-Proj"})
        method = cfg["Method"]
        n_ep_zci = cfg["N_EP"]
        cl_sites = site_meta.index[site_meta["Cluster"] == cl]

        zci_series = compute_zci(
            method,
            taxa_ra=taxa_relabd.loc[cl_sites],
            pollution_scores=site_meta.loc[cl_sites, "Pollution_Score"],
            n_ep=n_ep_zci,
            nmds_results={
                "coords_df": nmds_dict[cl].coords_df,
                "ref_label": nmds_dict[cl].ref_label,
                "deg_label": nmds_dict[cl].deg_label,
            },
        )

        corr = zci_correlation(
            zci_series,
            site_meta.loc[cl_sites, "Pollution_Score"],
        )

        zci_dict[cl] = ClusterZCI(
            cluster_id=cl,
            zci=zci_series,
            method=method,
            n_ep=n_ep_zci,
            r_pearson=corr["r_pearson"],
            p_pearson=corr["p_pearson"],
            r_spearman=corr["r_spearman"],
            p_spearman=corr["p_spearman"],
        )
        _log(f"       {zci_dict[cl].summary_line()}")

    # ── 8. Tables ────────────────────────────────────────────────────
    _log("[8/10] Saving tables …")

    # NMDS summary table
    nmds_rows = []
    for cl in clusters:
        nm = nmds_dict[cl]
        nmds_rows.append({
            "Cluster": cl,
            "n_sites": nm.n_sites,
            "Stress": nm.stress,
            "Var_NMDS1": nm.var_explained[0],
            "Var_NMDS2": nm.var_explained[1],
            "N_EP": nm.n_ep,
            "NMDS1_flipped": nm.flipped,
        })
    nmds_tbl = pd.DataFrame(nmds_rows).set_index("Cluster")
    save_table(nmds_tbl, tables_dir / "nmds_summary",
               formats=table_formats, verbose=verbose)

    # ZCI summary table
    zci_rows = []
    for cl in clusters:
        zc = zci_dict[cl]
        zci_rows.append({
            "Cluster": cl,
            "Method": zc.method,
            "N_EP": zc.n_ep,
            "ZCI_min": zc.zci.min(),
            "ZCI_max": zc.zci.max(),
            "ZCI_mean": zc.zci.mean(),
            "ZCI_std": zc.zci.std(),
            "r_Pearson": zc.r_pearson,
            "p_Pearson": zc.p_pearson,
            "r_Spearman": zc.r_spearman,
            "p_Spearman": zc.p_spearman,
            "Significance": zc.significance_stars,
        })
    zci_tbl = pd.DataFrame(zci_rows).set_index("Cluster")
    save_table(zci_tbl, tables_dir / "zci_summary",
               formats=table_formats, verbose=verbose)

    # ── 9. Figures ───────────────────────────────────────────────────
    if save_plots:
        _log("[9/10] Creating figures …")

        # NMDS biplot
        fig_biplot, _ = plot_nmds_biplot(
            nmds_dict, site_meta, clusters, p20, p80,
        )
        save_figure(fig_biplot, figures_dir / "nmds_biplot",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_biplot)

        # ZCI distribution
        fig_dist, _ = plot_zci_distribution(
            zci_dict, site_meta, clusters,
        )
        save_figure(fig_dist, figures_dir / "zci_distribution",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_dist)

        # ZCI vs Pollution Score
        fig_scatter, _ = plot_zci_vs_pollution(
            zci_dict, site_meta, clusters,
        )
        save_figure(fig_scatter, figures_dir / "zci_vs_pollution",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_scatter)

    # ── 10. Save augmented artifact ──────────────────────────────────
    _log("[10/10] Saving augmented artifact …")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Build multi-index columns matching project convention
    cols = pd.MultiIndex.from_tuples([
        ("04_bray_curtis_nmds", "raw", "NMDS1"),
        ("04_bray_curtis_nmds", "raw", "NMDS2"),
        ("04_bray_curtis_nmds", "raw", "ZCI"),
        ("04_bray_curtis_nmds", "raw", "ZCI_Method"),
    ])
    aug = pd.DataFrame(index=data.index, columns=cols)

    for cl in clusters:
        nm = nmds_dict[cl]
        zc = zci_dict[cl]
        real_sites = nm.real_site_ids
        coords = nm.coords_df.loc[real_sites]
        aug.loc[real_sites, ("04_bray_curtis_nmds", "raw", "NMDS1")] = \
            coords["NMDS1"].values
        aug.loc[real_sites, ("04_bray_curtis_nmds", "raw", "NMDS2")] = \
            coords["NMDS2"].values
        aug.loc[zc.zci.index, ("04_bray_curtis_nmds", "raw", "ZCI")] = \
            zc.zci.values
        aug.loc[zc.zci.index, ("04_bray_curtis_nmds", "raw", "ZCI_Method")] = \
            zc.method

    aug_path = artifacts_dir / "04_updated_data.xlsx"
    aug.to_excel(aug_path)
    _log(f"  ✓ Saved augmented data: {aug_path}")

    # ── Assemble result ──────────────────────────────────────────────
    result = NMDSPipelineResult(
        nmds_results=nmds_dict,
        zci_results=zci_dict,
        clusters=clusters,
        taxa_relabd=taxa_relabd,
        site_meta=site_meta,
        p20=p20,
        p80=p80,
    )

    _log(f"\n✓ NMDS pipeline complete.\n{result.summary()}")
    return result
