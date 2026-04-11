#!/usr/bin/env python3
"""Generate 3×2 ANOVA tables: (all / core-only / core+peripheral) × (env / taxa).

Reads the original study data and site-robustness labels, filters to
three site subsets, runs one-way ANOVA for environmental and taxa
variables, and writes 6 .xlsx files to WardsClustering/tables/.
"""

import sys
from pathlib import Path

# Allow imports from the project src directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from zci.core.anova import anova_table
from zci.io.readers import read_study_data, extract_block

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data/processed/complete_env_taxa_chemical_Feb_3.xlsx"
ROBUSTNESS_PATH = (
    PROJECT_ROOT
    / "results/02_taxa_assemblage/WardsClustering/tables/site_robustness.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "results/02_taxa_assemblage/WardsClustering/tables"

# ── Constants ────────────────────────────────────────────────────────────────
ENV_VARIABLES = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]

TAXA_COLUMNS = [
    "Oligochaeta", "Chironomidae", "Nematoda", "Dreissena", "Amphipoda",
    "Sphaeriidae", "Gastropoda", "Acari", "Hexagenia", "Caenis",
    "Hydrozoa", "Turbellaria", "Hydropsychidae", "Hirudinea",
    "Ceratopogonidae", "Other Trichoptera",
]

SUBSETS = {
    "all_sites": None,                       # no filtering
    "core_only": ["Core"],                   # keep Core only
    "core_peripheral": ["Core", "Peripheral"],  # drop Uncertain
}

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Read study data
    data = read_study_data(DATA_PATH)
    env_block = extract_block(data, "environmental", "raw")
    taxa_block = extract_block(data, "taxa", "raw")

    # 2. Read robustness (site → cluster + status)
    robustness = pd.read_excel(ROBUSTNESS_PATH, index_col=0)
    all_sites = robustness.index
    cluster_labels = robustness["Original_Cluster"]

    # Filter env/taxa to reference sites and drop NA
    env_vars_present = [v for v in ENV_VARIABLES if v in env_block.columns]
    taxa_vars_present = [t for t in TAXA_COLUMNS if t in taxa_block.columns]

    env_ref = env_block.loc[env_block.index.isin(all_sites), env_vars_present]
    taxa_ref = taxa_block.loc[taxa_block.index.isin(all_sites), taxa_vars_present]

    # Align to complete cases (no NaN in env)
    complete_idx = env_ref.dropna().index
    env_ref = env_ref.loc[complete_idx]
    taxa_ref = taxa_ref.loc[complete_idx]
    labels_all = cluster_labels.loc[complete_idx]

    # 3. Loop over subsets
    for subset_name, keep_statuses in SUBSETS.items():
        if keep_statuses is None:
            idx = labels_all.index
        else:
            mask = robustness.loc[labels_all.index, "Status"].isin(keep_statuses)
            idx = labels_all.index[mask]

        env_sub = env_ref.loc[idx]
        taxa_sub = taxa_ref.loc[idx]
        labels_sub = labels_all.loc[idx]

        n = len(idx)
        print(f"  {subset_name}: {n} sites")

        # env ANOVA
        env_anova = anova_table(
            env_sub, labels_sub, env_vars_present,
            transform="none", label_col="Variable",
        )
        out_env = OUTPUT_DIR / f"anova_env_{subset_name}.xlsx"
        env_anova.to_excel(out_env, index=False)
        print(f"    → {out_env.name}")

        # taxa ANOVA
        taxa_anova = anova_table(
            taxa_sub, labels_sub, taxa_vars_present,
            transform="none", label_col="Taxon",
        )
        out_taxa = OUTPUT_DIR / f"anova_taxa_{subset_name}.xlsx"
        taxa_anova.to_excel(out_taxa, index=False)
        print(f"    → {out_taxa.name}")

    print("\nDone – 6 ANOVA xlsx files written.")


if __name__ == "__main__":
    main()
