#!/usr/bin/env python
"""
Run Hindsight Cluster Relabel
==============================

Usage (from project root):
    python src2/run_hindsight_relabel.py

A lightweight post-Stage-2 script that remaps cluster labels according
to a user-defined mapping.  This is useful when the Ward-clustering
numbering does not match the desired ecological interpretation
(e.g. Cluster 3 from Ward should really be labelled Cluster 1).

Reads  : results2/02_taxa_assemblage/artifacts/02_updated_data.xlsx
Writes : results2/02_taxa_assemblage/artifacts/02_hindsight_updated_data.xlsx

The output file has exactly the same 3-level MultiIndex structure as
the original Stage 2 artifact; only the cluster values are replaced.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGE2_ARTIFACT = (
    PROJECT_ROOT / "results2" / "02_taxa_assemblage"
    / "artifacts" / "02_updated_data.xlsx"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "results2" / "02_taxa_assemblage"
    / "artifacts" / "02_hindsight_updated_data.xlsx"
)

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
    # 1: 3,
}


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


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    relabel_clusters(
        artifact_path=STAGE2_ARTIFACT,
        output_path=OUTPUT_PATH,
        label_map=LABEL_MAP,
    )
