# Contamination Assessment Module - Refactored Structure

## Overview

The contamination assessment module has been refactored into focused, single-responsibility modules for better maintainability and clarity.

## Module Structure

```
contamination_assessment/
├── __init__.py              # Package-level exports (use this for imports!)
├── pipeline.py              # High-level orchestration function
├── transformations.py       # Data transformation & preprocessing
├── pca_analysis.py          # PCA computation
├── visualizations.py        # Plotting functions (ridge plots, etc.)
├── validation.py            # RDA-based validation
├── scoring.py               # Pollution score computation
└── pca_pollution_scores.py  # DEPRECATED - kept for backward compatibility
```

## Recommended Import Patterns

### For the complete pipeline (recommended for most users):
```python
from zci.contamination_assessment import contamination_assessment_pipeline

results = contamination_assessment_pipeline(data)
```

### For individual functions:
```python
from zci.contamination_assessment import (
    log_z_score_transform,
    pca_with_PC_loadings,
    create_ridge_plot,
    compute_pollution_scores
)
```

### Old imports still work (backward compatibility):
```python
# These still work but are deprecated
from zci.contamination_assessment.pca_pollution_scores import log_z_score_transform
from zci.contamination_assessment.pca_pollution_scores import pca_with_PC_loadings
# etc.
```

## Module Responsibilities

### 1. `transformations.py`
- `log_z_score_transform()` - Apply log and z-score transformation
- `get_clustered_variable_order()` - Order variables by hierarchical clustering

### 2. `pca_analysis.py`
- `pca_with_PC_loadings()` - Perform PCA and extract components

### 3. `visualizations.py`
- `create_ridge_plot()` - Visualize PC loadings with ridge plot

### 4. `validation.py`
- `rda_pollution_species_analysis()` - RDA validation on reference sites

### 5. `scoring.py`
- `compute_pollution_scores()` - Calculate weighted pollution scores
- `merge_pollution_scores_into_data()` - Add scores to dataframes

### 6. `pipeline.py`
- `contamination_assessment_pipeline()` - Complete end-to-end workflow

## Benefits of Refactoring

1. **Clarity**: Each module has a single, clear purpose
2. **Maintainability**: Easier to find and update specific functionality
3. **Testability**: Individual components can be tested in isolation
4. **Reusability**: Functions can be imported and used independently
5. **Documentation**: Smaller modules are easier to document
6. **Backward Compatibility**: Existing code continues to work

## Migration Guide

### No changes required!
All existing code using `from zci.contamination_assessment.pca_pollution_scores import ...` will continue to work.

### Optional: Update to cleaner imports
For new code or when refactoring, use package-level imports:

**Old:**
```python
from zci.contamination_assessment.pca_pollution_scores import log_z_score_transform
from zci.contamination_assessment.pca_pollution_scores import pca_with_PC_loadings
from zci.contamination_assessment.pca_pollution_scores import compute_pollution_scores
```

**New (cleaner):**
```python
from zci.contamination_assessment import (
    log_z_score_transform,
    pca_with_PC_loadings,
    compute_pollution_scores
)
```

## Version History

- **v2.0.0**: Modularized structure (current)
- **v1.0.0**: Monolithic `pca_pollution_scores.py` (deprecated but still functional)
