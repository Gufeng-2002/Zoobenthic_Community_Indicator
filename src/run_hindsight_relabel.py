#!/usr/bin/env python
"""
Run Hindsight Cluster Relabel + ANOVA + Cluster Panel
======================================================

Usage (from project root):
    python src/run_hindsight_relabel.py

A post-Stage-2 script that:
  1. Remaps cluster labels according to a user-defined mapping.
  2. Runs ANOVA tests on environmental and taxa variables with the
     updated cluster labels.
  3. Produces the three-panel cluster figure (map + env bars + taxa bars).

Reads  : results/02_taxa_assemblage/artifacts/02_updated_data.xlsx
         data/processed/complete_env_taxa_chemical_Feb_3.xlsx
Writes : results/02_taxa_assemblage/artifacts/02_hindsight_updated_data.xlsx
         results/02_taxa_assemblage/tables/anova_env.xlsx
         results/02_taxa_assemblage/tables/anova_taxa.xlsx
         results/02_taxa_assemblage/figures/cluster_panel.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zci.io.readers import read_study_data, extract_block
from zci.io.writers import save_table, save_figure
from zci.core.anova import anova_table, extract_pvalues
from zci.core.transforms import octave_to_relative_abundance
from zci.viz.cluster_panel_plot import plot_cluster_panel, TAXA_DISPLAY_ORDER

# ── paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"

STAGE2_ARTIFACT = (
    PROJECT_ROOT / "results" / "02_taxa_assemblage"
    / "artifacts" / "02_updated_data.xlsx"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "02_taxa_assemblage"
    / "artifacts" / "02_hindsight_updated_data.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "02_taxa_assemblage"
MAPS_DIR   = PROJECT_ROOT / "data" / "maps"

# =====================================================================
#  >>>  EDIT THIS MAPPING  <<<
#
#  Keys   = original cluster labels (from Stage 2)
#  Values = desired new cluster labels
#
#  Example: {3: 1}  means  "every site that was Cluster 3 becomes Cluster 1"
#  Labels not mentioned in the mapping are left unchanged.
# =====================================================================

LABEL_MAP: dict[int, int] = {
    3: 1,
}

# ── Configuration for ANOVA / cluster panel ──────────────────────────

ENV_VARIABLES: list[str] = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]

ANOVA_TRANSFORM: str = "none"            # 'none', 'log', or 'boxcox'
FIGURE_FORMATS: tuple[str, ...] = ("png",)
TABLE_FORMATS: tuple[str, ...] = ("xlsx",)

# ── 16 study taxa (octave columns) ──────────────────────────────────

TAXA_COLUMNS: list[str] = [
    "Acari", "Amphipoda", "Caenis", "Ceratopogonidae",
    "Chironomidae", "Dreissena", "Gastropoda", "Hexagenia",
    "Hirudinea", "Hydropsychidae", "Hydrozoa", "Nematoda",
    "Oligochaeta", "Other Trichoptera", "Sphaeriidae", "Turbellaria",
]


# ── relabel logic ────────────────────────────────────────────────────


def relabel_clusters(
    artifact_path: str | Path,
    output_path: str | Path,
    label_map: dict[int, int],
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read a Stage 2 artifact, remap cluster labels, and save.

    Parameters
    ----------
    artifact_path : path
        ``02_updated_data.xlsx`` from Stage 2.
    output_path : path
        Where to write the relabelled artifact.
    label_map : dict
        ``{old_label: new_label}``.  Labels absent from the map are
        kept as-is.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        The relabelled artifact (same structure as the input).
    """
    artifact_path = Path(artifact_path)
    output_path = Path(output_path)

    if verbose:
        print("=" * 60)
        print("HINDSIGHT CLUSTER RELABEL")
        print("=" * 60)

    # ── 1. Read ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n[1/3] Reading Stage 2 artifact:\n      {artifact_path}")
    df = pd.read_excel(artifact_path, header=[0, 1, 2], index_col=0)

    # locate the Cluster column
    cluster_key = ("02_taxa_assemblage", "raw", "Cluster")
    if cluster_key not in df.columns:
        # fallback search
        cands = [c for c in df.columns if "Cluster" in str(c)]
        if not cands:
            raise ValueError("Cannot find a 'Cluster' column in the artifact.")
        cluster_key = cands[0]

    original = df[cluster_key].copy()
    n_total = len(original)
    n_labelled = original.notna().sum()

    if verbose:
        print(f"      {n_total} sites total, {n_labelled} with cluster labels")
        print(f"      Original distribution:")
        for g in sorted(original.dropna().unique()):
            print(f"        Cluster {int(g)}: {(original == g).sum()} sites")

    # ── 2. Remap ─────────────────────────────────────────────────────
    if verbose:
        print(f"\n[2/3] Applying label map: {label_map}")

    # Use a two-pass approach via temporary sentinels to handle swaps
    # (e.g. {1: 3, 3: 1}) without collisions.
    tmp = original.copy()
    sentinel_map: dict[int, float] = {}
    for i, (old, new) in enumerate(label_map.items()):
        sentinel = -(i + 1000)          # negative sentinel that can't collide
        sentinel_map[sentinel] = new
        tmp = tmp.replace({float(old): float(sentinel)})

    for sentinel, new in sentinel_map.items():
        tmp = tmp.replace({float(sentinel): float(new)})

    df[cluster_key] = tmp

    if verbose:
        print("      New distribution:")
        new_col = df[cluster_key]
        for g in sorted(new_col.dropna().unique()):
            print(f"        Cluster {int(g)}: {(new_col == g).sum()} sites")

    # ── 3. Save ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n[3/3] Saving relabelled artifact:\n      {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path)

    if verbose:
        print(f"\n✓ Hindsight relabel complete.")

    return df


def run_anova_and_panel(
    data_path: str | Path,
    relabelled_artifact: pd.DataFrame,
    output_dir: str | Path,
    maps_dir: str | Path,
    *,
    env_variables: list[str] | None = None,
    anova_transform: str = "none",
    taxa_columns: list[str] | None = None,
    map_func=None,
    figure_formats: tuple[str, ...] = ("png",),
    table_formats: tuple[str, ...] = ("xlsx",),
    verbose: bool = True,
) -> None:
    """Run ANOVA tests and produce cluster panel figure on relabelled clusters.

    Parameters
    ----------
    data_path : path
        Path to the original study-data Excel file.
    relabelled_artifact : pd.DataFrame
        The relabelled artifact DataFrame (3-level MultiIndex columns).
    output_dir : path
        Root output directory (``results/02_taxa_assemblage``).
    maps_dir : path
        Path to ``data/maps/`` folder with shapefiles.
    env_variables : list of str
        Environmental variable names for ANOVA / bar plot.
    anova_transform : str
        Pre-ANOVA transformation: ``"none"``, ``"log"``, ``"boxcox"``.
    taxa_columns : list of str
        Taxa column names.
    figure_formats / table_formats : tuple of str
        File formats for outputs.
    verbose : bool
        Print progress messages.
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    if env_variables is None:
        env_variables = ENV_VARIABLES
    if taxa_columns is None:
        taxa_columns = TAXA_COLUMNS

    _log("\n" + "=" * 60)
    _log("ANOVA & CLUSTER PANEL (on relabelled clusters)")
    _log("=" * 60)

    # ── Extract cluster labels for reference sites ───────────────────
    cluster_key = ("02_taxa_assemblage", "raw", "Cluster")
    if cluster_key not in relabelled_artifact.columns:
        cands = [c for c in relabelled_artifact.columns if "Cluster" in str(c)]
        if not cands:
            raise ValueError("Cannot find a 'Cluster' column in the artifact.")
        cluster_key = cands[0]

    all_clusters = relabelled_artifact[cluster_key]
    ref_mask = all_clusters.notna()
    labels_ref = all_clusters.loc[ref_mask].astype(int)
    labels_ref.name = "Cluster"

    _log(f"\n      Reference sites: {ref_mask.sum()}")
    _log(f"      Cluster distribution (after relabel):")
    for g in sorted(labels_ref.unique()):
        _log(f"        Cluster {g}: {(labels_ref == g).sum()} sites")

    # ── Read original data for env and taxa blocks ───────────────────
    _log(f"\n[1/3] Reading original study data for env/taxa blocks …")
    data = read_study_data(data_path)

    # ── Environmental variables ──────────────────────────────────────
    env_ref = extract_block(data, "environmental", "raw")
    env_vars_present = [v for v in env_variables if v in env_ref.columns]
    env_ref = env_ref.loc[ref_mask, env_vars_present]

    # ── Taxa variables (octave) ──────────────────────────────────────
    taxa_all = extract_block(data, "taxa", "raw")[list(taxa_columns)]
    taxa_ref = taxa_all.loc[ref_mask]

    # ── ANOVA — environmental variables ──────────────────────────────
    _log(f"\n[2/3] ANOVA on {len(env_vars_present)} environmental variables …")
    env_anova = anova_table(
        env_ref, labels_ref, env_vars_present,
        transform=anova_transform, label_col="Variable",
    )
    save_table(env_anova, tables_dir / "anova_env",
               formats=table_formats, verbose=verbose)
    env_pvals = extract_pvalues(env_anova, label_col="Variable")

    # ── ANOVA — taxa variables (octave scale) ────────────────────────
    _log(f"      ANOVA on {taxa_ref.shape[1]} taxa (octave scale) …")
    taxa_anova = anova_table(
        taxa_ref, labels_ref, list(taxa_ref.columns),
        transform=anova_transform, label_col="Taxon",
    )
    save_table(taxa_anova, tables_dir / "anova_taxa",
               formats=table_formats, verbose=verbose)
    taxa_pvals = extract_pvalues(taxa_anova, label_col="Taxon")

    # ── Three-panel cluster figure ───────────────────────────────────
    _log(f"\n[3/3] Saving cluster panel figure …")
    sample_info = extract_block(data, "sample_info", "raw")
    lat = sample_info.loc[ref_mask, "Latitude"]
    lon = sample_info.loc[ref_mask, "Longitude"]

    # relative-abundance version for the taxa bar chart
    taxa_relabd = octave_to_relative_abundance(taxa_ref)

    fig_panel, _ = plot_cluster_panel(
        cluster_labels=labels_ref,
        lat=lat,
        lon=lon,
        env_data=env_ref,
        taxa_octave=taxa_ref,
        taxa_relabd=taxa_relabd,
        env_pvalues=env_pvals,
        taxa_pvalues=taxa_pvals,
        maps_dir=maps_dir,
        env_vars=env_vars_present,
        map_func=map_func,
        taxa_order=TAXA_DISPLAY_ORDER,
    )
    save_figure(fig_panel, figures_dir / "cluster_panel",
                formats=figure_formats, verbose=verbose)
    plt.close(fig_panel)

    _log(f"\n✓ ANOVA & cluster panel complete.")


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Relabel clusters
    relabelled_df = relabel_clusters(
        artifact_path=STAGE2_ARTIFACT,
        output_path=OUTPUT_PATH,
        label_map=LABEL_MAP,
    )

    # Step 2: Run ANOVA + cluster panel on the relabelled clusters
    run_anova_and_panel(
        data_path=DATA_PATH,
        relabelled_artifact=relabelled_df,
        output_dir=OUTPUT_DIR,
        maps_dir=MAPS_DIR,
        env_variables=ENV_VARIABLES,
        anova_transform=ANOVA_TRANSFORM,
        figure_formats=FIGURE_FORMATS,
        table_formats=TABLE_FORMATS,
    )
