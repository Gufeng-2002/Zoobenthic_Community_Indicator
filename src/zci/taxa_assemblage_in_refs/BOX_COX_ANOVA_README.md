# Box-Cox Transformation and ANOVA Analysis

This module provides Box-Cox transformation followed by one-way ANOVA to identify environmental variables and taxa that differ significantly across clusters.

## Overview

The Box-Cox transformation is applied to stabilize variance and normalize distributions before performing ANOVA, which improves the validity of statistical tests when comparing means across groups.

## Key Functions

### `perform_boxcox_anova_analysis()`

Complete analysis combining both environmental and taxa variables:

```python
from zci.taxa_assemblage_in_refs import perform_boxcox_anova_analysis

results = perform_boxcox_anova_analysis(
    raw_data=raw_data,
    multiindex_data=multiindex_data,
    cluster_column='clusters',
    env_variables=['Measured Depth (m)', 'Velocity  at bottom (m/sec)_Imputed', 
                   'Water DO Bottom (mg/L)', 'Temperature (oC)', 
                   'MPS (Phi)', 'LOI (%)'],
    taxa_transformation='hellinger',
    top_n_taxa=16,
    verbose=True
)
```

**Returns:**
- `env_results`: Environmental variables ANOVA results
- `taxa_results`: Taxa variables ANOVA results  
- `summary`: Overall summary statistics

### `boxcox_transform_and_anova_env()`

Analyze environmental variables only:

```python
from zci.taxa_assemblage_in_refs import boxcox_transform_and_anova_env

env_results = boxcox_transform_and_anova_env(
    raw_data=reference_sites_data,
    cluster_column='clusters',
    env_variables=['Depth', 'Temperature', 'DO'],
    verbose=True
)
```

### `boxcox_transform_and_anova_taxa()`

Analyze taxa variables only:

```python
from zci.taxa_assemblage_in_refs import boxcox_transform_and_anova_taxa

taxa_results = boxcox_transform_and_anova_taxa(
    raw_data=raw_data,
    multiindex_data=multiindex_data,
    cluster_column='clusters',
    taxa_transformation='hellinger',
    top_n_taxa=16,
    verbose=True
)
```

## Integration with Pipeline

The Box-Cox ANOVA analysis can be run as part of the complete pipeline:

```python
from zci.taxa_assemblage_in_refs import reference_sites_taxa_assemblage_pipeline

results = reference_sites_taxa_assemblage_pipeline(
    raw_data=raw_data,
    multiindex_data=multiindex_data,
    pollution_column='Pollution_Score',
    reference_percentile=50,
    species_transformation='hellinger',
    n_clusters=3,
    run_boxcox_anova=True,              # Enable ANOVA
    anova_taxa_transformation='hellinger',
    anova_top_n_taxa=None,              # Analyze all taxa
    verbose=True
)

# Access ANOVA results
anova_results = results['anova_results']
env_significant = anova_results['env_results']['n_significant']
taxa_significant = anova_results['taxa_results']['n_significant']
```

## Parameters

### Taxa Transformation Options

Before Box-Cox transformation, taxa can be transformed using:
- `'hellinger'`: Square root of relative abundances (default)
- `'chord'`: Unit-length normalization
- `'octave'`: Log2 transformation of proportions
- `'none'`: No initial transformation

### Key Parameters

- **env_variables**: List of environmental variable names to analyze
- **taxa_transformation**: Initial transformation for taxa ('hellinger', 'chord', 'octave', 'none')
- **top_n_taxa**: Number of most abundant taxa to analyze (None = all taxa)
- **verbose**: Print detailed progress and results

## Output Structure

### Environmental Variables Results

```python
env_results = {
    'anova_results': pd.DataFrame,      # F-statistic, p-value, significance for each variable
    'transformation_info': dict,        # Box-Cox parameters (lambda, shift, interpretation)
    'transformed_data': dict,           # Transformed Series for each variable
    'n_significant': int                # Count of significant variables
}
```

### Taxa Results

```python
taxa_results = {
    'anova_results': pd.DataFrame,      # F-statistic, p-value, significance for each taxon
    'transformation_info': dict,        # Box-Cox parameters for each taxon
    'transformed_data': dict,           # Transformed Series for each taxon
    'n_significant': int,               # Count of significant taxa
    'significant_taxa': list,           # Names of significant taxa
    'taxa_transformation_used': str     # Transformation method used
}
```

## Example Output

```
================================================================================
BOX-COX TRANSFORMATION + ANOVA: Environmental Variables
================================================================================

Analyzing 6 environmental variables
Sites: 245 (across 3 clusters)
Cluster distribution: {0: 78, 1: 98, 2: 69}

--------------------------------------------------------------------------------
ANOVA Results (sorted by p-value):
--------------------------------------------------------------------------------
                 Variable  F-statistic   p-value  Significance  Lambda  ...
         MPS (Phi)           45.23      0.0000        ***       0.234   
  Measured Depth (m)         32.15      0.0000        ***       0.512   
       LOI (%)               18.43      0.0001        ***      -0.123   
   Temperature (oC)           2.34      0.0982         ns       0.891   

Summary:
  - Total variables: 6
  - Significant differences (p < 0.05): 4
  - Non-significant: 2
```

## Statistical Notes

1. **Box-Cox Transformation**: Automatically finds optimal λ parameter to normalize each variable
2. **ANOVA Assumptions Checked**:
   - Normality (Shapiro-Wilk test)
   - Homogeneity of variance (Levene's test)
3. **Significance Levels**:
   - `***`: p < 0.001
   - `**`: p < 0.01
   - `*`: p < 0.05
   - `ns`: p ≥ 0.05

## References

Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. *Journal of the Royal Statistical Society: Series B*, 26(2), 211-252.
