# Zoobenthic Community-Condition Index (ZCI) -- Project Code

> **Feng Gu's Master's Thesis** -- *"Zoobenthic Community Indicator of Sediment Contamination"*

This repository implements a **7-step reproducible pipeline** (5 main stages + 2 auxiliary stages)
that quantifies how benthic macroinvertebrate communities respond to sediment contamination
across the **St. Clair--Detroit River System (SCDRS)**.
A separate Detroit River (DR)-only variant mirrors the same structure.

---

## Framework Overview

The central idea is a sequential chain: each stage produces an **artifact**
(an augmented Excel file) consumed by the next stage.

```
Stage 1 --> RDA --> Stage 2 --> Hindsight Relabel --> Stage 3 --> Stage 4 --> Stage 5
```

### Main Stages

| # | Stage | What it does |
|---|-------|-------------|
| **1** | **Pollution PCA** | PCA on sediment-chemistry variables to derive a composite **Pollution Score** per site. Sites below the 20th-percentile threshold are flagged as **reference**. |
| **2** | **Taxa Assemblage Clustering** | Ward hierarchical clustering on octave-transformed taxa abundances of **reference sites only** to identify distinct biological assemblage groups. |
| **3** | **LDA Classification** | Linear Discriminant Analysis trained on reference-site environmental variables and cluster labels. Monte Carlo cross-validation (1000 iterations). Predicts cluster membership for **all non-reference sites**, extending the classification to the full study area. |
| **4** | **Bray-Curtis NMDS + ZCI** | Bray-Curtis dissimilarity followed by 2-D NMDS ordination. Constructs a **Zoobenthos Community-Condition Index (ZCI)** per cluster, measuring how far each site's community has shifted from the reference condition. |
| **5** | **Piecewise Quantile Regression** | Segmented quantile regressions of ZCI vs Pollution Score per cluster. Detects optimal breakpoints, builds 90% wild-bootstrap CIs, and runs sample-size sensitivity analysis. |

### Auxiliary Stages

| Stage | Position in chain | What it does |
|-------|-------------------|-------------|
| **RDA** (Redundancy Analysis) | After Stage 1, before Stage 2 | Fits RDA with taxa as response and environmental variables as predictors on reference sites. Tests significance via 999 permutations. Produces a triplot coloured by cluster labels. |
| **Hindsight Relabel** | After Stage 2, before Stage 3 | Post-clustering correction that remaps cluster labels (e.g. merges the smallest cluster into another). Runs ANOVA on environmental and taxa variables by new labels, then produces a three-panel cluster figure (map + environmental bars + taxa bars). |

### Pipeline Runners

| Script | Scope |
|--------|-------|
| `src/run_SCDRS_full_pipeline.py` | Full SCDRS pipeline (all waterbodies, 233 sites) |
| `src/run_DR_full_pipeline.py` | Detroit River only (146 sites, adds velocity as an extra env variable) |
| `src/run_stage1.py` ... `run_stage5.py` | Individual stage runners |
| `src/run_stage_rda.py` | Standalone RDA runner |
| `src/run_hindsight_relabel.py` | Standalone hindsight relabel runner |

---

## Results Structure

Every stage writes to a dedicated subfolder with a consistent `artifacts/`, `figures/`, `tables/` layout.

```
results/
|
+-- 01_pollution_assessment/          <-- Stage 1
|   +-- artifacts/
|   |   +-- 01_updated_data.xlsx
|   +-- figures/
|   |   +-- variance_explained.png
|   |   +-- ridge_loadings.png
|   |   +-- corridor_bifurcation.png
|   +-- tables/
|       +-- pc_loadings.xlsx
|       +-- site_scores.xlsx
|
+-- 02_taxa_assemblage/               <-- Stage 2 + Hindsight Relabel
|   +-- artifacts/
|   |   +-- 02_updated_data.xlsx
|   |   +-- 02_hindsight_updated_data.xlsx
|   +-- figures/
|   |   +-- ward_dendrogram.png
|   |   +-- cluster_panel.png
|   +-- tables/
|       +-- reference_taxa_clusters.xlsx
|       +-- anova_env.xlsx
|       +-- anova_taxa.xlsx
|
+-- 03_LDA_Classification/            <-- Stage 3
|   +-- artifacts/
|   |   +-- 03_updated_data.xlsx
|   +-- figures/
|   |   +-- lda_triplot.png
|   |   +-- cluster_comparison.png
|   +-- tables/
|       +-- lda_axes_summary.xlsx
|       +-- lda_classification_report.xlsx
|       +-- lda_confusion_matrix.xlsx
|       +-- lda_env_significance.xlsx
|       +-- mccv_classification_report.xlsx
|       +-- mccv_confusion_matrix.xlsx
|
+-- 04_bray_curtis_NMDS/              <-- Stage 4
|   +-- artifacts/
|   |   +-- 04_updated_data.xlsx
|   +-- figures/
|   |   +-- nmds_biplot.png
|   |   +-- zci_distribution.png
|   |   +-- zci_vs_pollution.png
|   +-- tables/
|       +-- nmds_summary.xlsx
|       +-- zci_summary.xlsx
|
+-- 05_piecewise_qr/                  <-- Stage 5
|   +-- artifacts/
|   |   +-- 05_updated_data.xlsx
|   +-- figures/
|   |   +-- qr_ci_errorbars_cluster_*.png
|   |   +-- qr_three_quantiles_cluster_*.png
|   |   +-- sensitivity_cluster_*.png
|   |   +-- sensitivity_coverage_cluster_*.png
|   +-- tables/
|       +-- qr_coefficients_cluster_*.xlsx
|       +-- sensitivity_cluster_*.xlsx
|
+-- RDA_analysis/                     <-- RDA (auxiliary)
|   +-- figures/
|   |   +-- rda_triplot.png
|   +-- tables/
|       +-- rda_axes_summary.xlsx
|       +-- rda_terms_summary.xlsx
|
+-- DR_results/                       <-- Detroit River pipeline (mirrors above)
    +-- 01_pollution_assessment/
    +-- 02_taxa_assemblage/
    +-- 03_LDA_Classification/
    +-- 04_bray_curtis_NMDS/
    +-- 05_piecewise_qr/
    +-- RDA_analysis/
```

---

## Source Package

```
src/zci/
+-- core/       Statistical algorithms (PCA, clustering, LDA, NMDS, RDA, quantile regression, ZCI)
+-- io/         Data readers and writers
+-- models/     Result dataclasses for each stage
+-- pipeline/   High-level orchestration (one module per stage)
+-- viz/        Plotting functions (maps, PCA, clustering, LDA, NMDS, RDA, quantile regression)
```
