#!/usr/bin/env python
"""
Sweep N_REFERENCE_SITES from 40 to 70 and report pvclust AU values
for the 3 Ward clusters at each N.
"""
import pandas as pd
from pathlib import Path

from zci.io.readers import read_study_data, extract_block
from zci.core.clustering import ward_cluster, select_reference_sites
from zci.core.transforms import octave_transform
from zci.core.ward_robustness import run_pvclust

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Apr17.xlsx"
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment"
    / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
)

# ── Load data once ────────────────────────────────────────────────────
print("Loading data ...")
data = read_study_data(DATA_PATH)
stage1 = pd.read_excel(STAGE1_ARTIFACT, header=[0, 1, 2], index_col=0)
score_cols = [
    c for c in stage1.columns
    if c[0] == "01_pollution_assessment" and c[1] == "raw" and c[2].endswith("_Score")
]
pollution_score = stage1.loc[:, score_cols[0]]
taxa_all = extract_block(data, "taxa", "raw")
taxa_all = taxa_all.loc[taxa_all.index.intersection(pollution_score.index)]

N_CLUSTERS = 3
NBOOT = 1000
N_VALUES = list(range(40, 71))

# ── Sweep ─────────────────────────────────────────────────────────────
rows = []
for n_ref in N_VALUES:
    print(f"\n{'='*60}")
    print(f"  N_REFERENCE_SITES = {n_ref}")
    print(f"{'='*60}")

    ref_mask = select_reference_sites(pollution_score, quantile=n_ref)
    actual_n = ref_mask.sum()
    taxa_ref = taxa_all.loc[ref_mask]
    taxa_transformed = octave_transform(taxa_ref)

    labels_ref, Z = ward_cluster(taxa_transformed, n_clusters=N_CLUSTERS)
    cluster_sizes = labels_ref.value_counts().sort_index()
    print(f"  Actual ref sites: {actual_n}")
    print(f"  Cluster sizes: {dict(cluster_sizes)}")

    try:
        pvclust_df = run_pvclust(taxa_transformed, n_clusters=N_CLUSTERS, nboot=NBOOT, verbose=False)
        row = {"N": n_ref, "actual_n": actual_n}
        for _, r in pvclust_df.iterrows():
            k = int(r["Cluster"])
            row[f"AU_C{k}"] = round(r["AU"], 4)
            row[f"BP_C{k}"] = round(r["BP"], 4)
            row[f"size_C{k}"] = int(cluster_sizes.get(k, 0))
        rows.append(row)
        print(f"  AU: C1={row.get('AU_C1','?')}, C2={row.get('AU_C2','?')}, C3={row.get('AU_C3','?')}")
    except Exception as e:
        print(f"  ERROR: {e}")
        rows.append({"N": n_ref, "actual_n": actual_n, "error": str(e)})

# ── Results table ─────────────────────────────────────────────────────
results = pd.DataFrame(rows)
print("\n" + "=" * 80)
print("  SWEEP RESULTS: AU by N_REFERENCE_SITES")
print("=" * 80)
print(results.to_string(index=False))

# Save to file
out_path = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "WardsClustering" / "pvclust_AU_sweep_N40_70.xlsx"
out_path.parent.mkdir(parents=True, exist_ok=True)
results.to_excel(out_path, index=False)
print(f"\nSaved to: {out_path}")
