# Contamination Assessment Module - Architecture

## Module Organization

```
zci/contamination_assessment/
│
├── __init__.py                 # Package interface (USE THIS for imports!)
│   └── Exports all public functions from sub-modules
│
├── pipeline.py                 # 🎯 Main Entry Point
│   └── contamination_assessment_pipeline()
│       ├── Orchestrates complete workflow
│       ├── Calls functions from all modules below
│       └── Returns comprehensive results dict
│
├── transformations.py          # 🔄 Data Preprocessing
│   ├── log_z_score_transform()
│   │   ├── Log transformation (log1p)
│   │   └── Z-score standardization
│   └── get_clustered_variable_order()
│       └── Hierarchical clustering of variables
│
├── pca_analysis.py             # 📊 Dimensionality Reduction
│   ├── pca_with_PC_loadings()
│   │   ├── Fits PCA model
│   │   ├── Extracts loadings & scores
│   │   └── Optional visualization
│   └── _plot_pca_variance()
│       └── Internal: creates variance plots
│
├── visualizations.py           # 📈 Plotting
│   └── create_ridge_plot()
│       ├── Creates ridge plot of PC loadings
│       ├── Clusters variables for display
│       └── Saves to results/figures/
│
├── validation.py               # ✅ Biological Validation
│   ├── rda_pollution_species_analysis()
│   │   ├── Hellinger transformation of taxa
│   │   ├── Fits RDA model on reference sites
│   │   ├── Identifies biologically relevant PCs
│   │   └── Creates triplot visualization
│   └── _create_rda_triplot()
│       └── Internal: generates RDA triplot
│
├── scoring.py                  # 🎯 Score Computation
│   ├── compute_pollution_scores()
│   │   ├── Weighted sum of PC scores
│   │   ├── Default weights from RDA
│   │   └── Custom weights supported
│   └── merge_pollution_scores_into_data()
│       ├── Adds scores to raw dataframe
│       └── Adds scores to multi-index dataframe
│
├── pca_pollution_scores.py     # ⚠️ DEPRECATED
│   └── Compatibility wrapper
│       ├── Shows deprecation warning
│       └── Re-exports from new modules
│
└── README.md                   # 📖 Documentation
    ├── Module structure
    ├── Import patterns
    └── Migration guide
```

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  contamination_assessment_pipeline()                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Data Preparation                                  │  │
│  │    • Extract pollution block: data[("chemical","raw")]│ │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. Transformation (transformations.py)                │  │
│  │    • log_z_score_transform()                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3. Variable Ordering (transformations.py)             │  │
│  │    • get_clustered_variable_order() [if visualize]   │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 4. PCA Analysis (pca_analysis.py)                     │  │
│  │    • pca_with_PC_loadings()                          │  │
│  │    • Returns: PC_loadings, PC_scores                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 5. Visualization (visualizations.py)                  │  │
│  │    • create_ridge_plot() [if visualize]              │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 6. Validation (validation.py)                         │  │
│  │    • rda_pollution_species_analysis() [if enabled]   │  │
│  │    • Assesses biological relevance of PCs            │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 7. Scoring (scoring.py)                               │  │
│  │    • compute_pollution_scores()                      │  │
│  │    • merge_pollution_scores_into_data()              │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Returns: {                                            │  │
│  │   'raw_data': updated with Pollution_Score           │  │
│  │   'multiindex_data': updated with scores             │  │
│  │   'PC_loadings': component loadings                  │  │
│  │   'PC_scores': component scores                      │  │
│  │   'pollution_scores': final scores                   │  │
│  │   ... and more ...                                    │  │
│  │ }                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Import Patterns by Use Case

### Use Case 1: Complete Pipeline (Most Common)
```python
from zci.contamination_assessment import contamination_assessment_pipeline

results = contamination_assessment_pipeline(
    data,
    visualize=True,
    run_rda_validation=True
)
```

### Use Case 2: Step-by-Step Analysis
```python
from zci.contamination_assessment import (
    log_z_score_transform,
    pca_with_PC_loadings,
    compute_pollution_scores
)

# Manual step-by-step
transformed = log_z_score_transform(pollution_data)
PC_loadings, PC_scores = pca_with_PC_loadings(transformed)
scores = compute_pollution_scores(PC_scores)
```

### Use Case 3: Custom Weighting Schemes
```python
from zci.contamination_assessment import (
    contamination_assessment_pipeline
)

custom_weights = {'PC1': 2.0, 'PC2': 1.5, 'PC3': 3.0}
results = contamination_assessment_pipeline(
    data,
    pc_weights=custom_weights,
    visualize=False  # Speed up for batch processing
)
```

## Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Loose Coupling**: Modules can be used independently
3. **High Cohesion**: Related functions are grouped together
4. **Backward Compatibility**: Old imports still work (with deprecation warning)
5. **Progressive Disclosure**: Simple interface (pipeline) for common cases,
   detailed control available when needed
