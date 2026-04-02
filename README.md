# Zoobenthic Community-Condition Index (ZCI) -- Project Code

> **Feng Gu's Master's Thesis** -- *"Zoobenthic Community Indicator of Sediment Contamination"*

This repository implements a reproducible benthic-community analysis workflow for the **St. Clair--Detroit River System (SCDRS)**, plus a Detroit River-only variant. The current codebase is organized around a **small set of top-level runners** and a refactored `zci` package with clear layers.

---

## Framework Overview

The workflow is still artifact-driven: each major step writes an augmented workbook that can be consumed by later steps.

```text
Stage 1 --> optional RDA / threshold sensitivity --> Stage 2 (choose method)
        --> Stage 3 (NMDS + ZCI) --> Stage 4 (piecewise QR)
```

### Main Stages

| # | Stage | What it does |
| --- | --- | --- |
| **1** | **Pollution Assessment** | PCA on sediment chemistry to derive site-level contamination scores and define reference sites. The stage also supports threshold-sensitivity analysis across alternative cutoffs. |
| **2** | **Taxa Assemblage** | Assigns assemblage groups using one of two implementations: **Ward + LDA** or **MRT**. Both operate from Stage 1 outputs and write their own method-specific results folders. |
| **3** | **Bray-Curtis NMDS + ZCI** | Builds a Bray-Curtis dissimilarity matrix, fits 2-D NMDS, and computes cluster-wise ZCI values using configurable endpoint and scoring methods. |
| **4** | **Piecewise Quantile Regression** | Fits segmented quantile regressions of ZCI against pollution score, adds wild-bootstrap confidence intervals, and supports sample-size sensitivity analysis. |

### Auxiliary Analyses

| Analysis | Where it fits | What it does |
| --- | --- | --- |
| **RDA** | After Stage 1 | Redundancy analysis on reference sites, with permutation testing and triplots. |
| **Threshold Sensitivity** | Inside Stage 1 | Sweeps alternative reference cutoffs and compares downstream ecological signal. |
| **LDA Classification** | Inside the Ward + LDA route | Trains on reference-site labels and predicts assemblages for non-reference sites. |
| **Relabelling / ANOVA / Cluster Panel** | Method-specific post-processing | Used where needed to remap cluster labels, compare groups, and generate summary figures. |

---

## Top-Level Runners

| Script | Role |
| --- | --- |
| `src/run_SCDRS_full_pipeline.py` | Full SCDRS workflow across the main stages and auxiliary analyses. |
| `src/run_DR_full_pipeline.py` | Detroit River-only workflow with DR filtering and an extra velocity variable. |
| `src/run_stage1.py` | Stage 1 only: pollution assessment plus threshold sensitivity. |
| `src/run_stage2_WardsLDA.py` | Stage 2 using the Ward clustering + LDA path. |
| `src/run_stage2_MRT.py` | Stage 2 using the multivariate regression tree path. |
| `src/run_stage3.py` | Stage 3 only: Bray-Curtis NMDS + ZCI. |
| `src/run_stage4.py` | Stage 4 only: piecewise quantile regression. |
| `src/compare_mrt_transforms.py` | Utility script for comparing alternative taxa transforms in the MRT workflow. |

### Stage 2 Branches

| Method | Pipeline module | Notes |
| --- | --- | --- |
| **Ward + LDA** | `src/zci/pipeline/taxa_assemblage.py` | Clusters reference sites, then uses LDA to extend labels to non-reference sites. |
| **MRT** | `src/zci/pipeline/mrt.py` | Fits a multivariate regression tree on reference sites and predicts assemblages through the tree-based route. |

---

## Results Structure

Outputs are grouped by stage, and most stage folders follow the same `artifacts/`, `figures/`, `tables/` pattern.

```text
results/
|
+-- 01_pollution_assessment/
|   +-- contamination_stressors/
|   +-- cutoff_reference/
|
+-- 02_taxa_assemblage/
|   +-- MRT_Method/
|   +-- Wards_LDA/
|   +-- reproduction_with_same_taxa_data/
|
+-- 03_bray_curtis_NMDS/
|   +-- artifacts/
|   +-- figures/
|   +-- tables/
|
+-- 05_piecewise_qr/
|   +-- artifacts/
|   +-- figures/
|   +-- latex_tables/
|   +-- tables/
|
+-- DR_results/
|   +-- 01_pollution_assessment/
|   +-- 02_taxa_assemblage/
|   +-- 03_LDA_Classification/
|   +-- 04_bray_curtis_NMDS/
|   +-- 05_piecewise_qr/
|   +-- RDA_analysis/
|
+-- ref_threshold_sensitivity/
|   +-- figures/
|   +-- latex_tables/
|   +-- tables/
```

---

## Source Package

The refactored package is intentionally layered.

| Package | Responsibility |
| --- | --- |
| `src/zci/io/` | Read raw workbooks and write stage outputs. |
| `src/zci/core/` | Statistical methods and transforms: PCA, clustering, MRT, LDA, RDA, NMDS, ZCI scoring, piecewise QR, ANOVA, threshold sensitivity. |
| `src/zci/models/` | Lightweight result dataclasses returned by the core and pipeline layers. |
| `src/zci/pipeline/` | High-level orchestration modules such as `pollution_assessment`, `taxa_assemblage`, `mrt`, `bray_curtis_nmds`, `piecewise_qr_pipeline`, `rda_analysis`, and `threshold_sensitivity`. |
| `src/zci/viz/` | Stage-specific plotting helpers for PCA, clustering, MRT, LDA, RDA, NMDS, threshold sensitivity, maps, and QR figures. |

In short: `run_*.py` scripts are now thin entry points, while `src/zci/` holds the actual implementation.
