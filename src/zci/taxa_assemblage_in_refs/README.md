# Taxa Assemblage in Reference Sites Module

## Overview

This module provides a comprehensive pipeline for analyzing taxa assemblages at reference sites in the St. Clair-Detroit River System. It performs velocity imputation, reference site selection based on pollution scores, hierarchical clustering of taxa composition, and generates publication-quality visualizations.

## Quick Start

```python
from zci.taxa_assemblage_in_refs import reference_sites_taxa_assemblage_pipeline

# Run the complete pipeline with default parameters
results = reference_sites_taxa_assemblage_pipeline(
    raw_data=raw_data,
    multiindex_data=multiindex_data
)

# Access the outputs
updated_raw_data = results['raw_data']
updated_multiindex = results['multiindex_data']
cluster_labels = results['cluster_labels']
visualization = results['cluster_visualization_fig']
```

## Pipeline Architecture

The pipeline consists of 6 main steps:

1. **Velocity Imputation**: Fills missing velocity values using spatial relationships
2. **Reference Site Selection**: Selects least-polluted sites based on percentile threshold
3. **Taxa Data Extraction**: Extracts and prepares species composition data
4. **Hierarchical Clustering**: Performs clustering with configurable transformation and linkage
5. **Label Assignment**: Adds cluster labels to datasets (NaN for non-reference sites)
6. **Visualization**: Creates comprehensive 3-panel figures

## Main Function: `reference_sites_taxa_assemblage_pipeline`

### Parameters

#### Required
- **`raw_data`** (pd.DataFrame): Raw data with single-level columns
- **`multiindex_data`** (pd.DataFrame): Multi-index data with hierarchical columns

#### Key Configuration Parameters
- **`pollution_column`** (str, default='Pollution_Score'): Column name for pollution scores
- **`reference_percentile`** (float, default=52): Percentile for reference site selection (bottom p%)
- **`species_transformation`** (str, default='hellinger'): Taxa transformation method
  - Options: 'hellinger', 'chord', 'octave', 'none'
- **`linkage_method`** (str, default='ward'): Hierarchical clustering linkage
  - Options: 'ward', 'average', 'complete', 'single'
- **`n_clusters`** (int, default=3): Number of clusters to create

#### Visualization Parameters
- **`top_n_taxa`** (int, default=15): Number of most abundant taxa to display
- **`env_variables`** (list, optional): Environmental variables to visualize
- **`create_cluster_visualization`** (bool, default=True): Generate final 3-panel figure

#### Diagnostic Parameters
- **`create_dendrogram_comparison`** (bool, default=False): 4-method dendrogram comparison
- **`create_comparison_plots`** (bool, default=False): Linkage method comparison
- **`create_fusion_plots`** (bool, default=False): Fusion level analysis
- **`create_ward_analysis`** (bool, default=False): Ward's method detailed analysis
- **`label_positions`** (list, optional): Manual cluster label positions for Ward plot

#### Other
- **`verbose`** (bool, default=True): Print progress messages

### Returns

Dictionary containing:
- **`raw_data`**: Updated DataFrame with 'if_ref' and 'clusters' columns
- **`multiindex_data`**: Updated multi-index DataFrame with cluster labels
- **`cluster_labels`**: Series with cluster assignments for reference sites
- **`reference_sites_count`**: Number of reference sites selected
- **`cluster_distribution`**: Distribution of sites across clusters
- **`velocity_imputation_fig`**: Velocity imputation visualization
- **`clustering_results`**: Complete clustering results with linkage matrix and figures
- **`cluster_visualization_fig`**: Final 3-panel visualization
- **`habitat_comparison`**: (optional) Habitat variable comparison results

## Usage Examples

### Example 1: Basic Usage with Defaults

```python
from zci.taxa_assemblage_in_refs import reference_sites_taxa_assemblage_pipeline

results = reference_sites_taxa_assemblage_pipeline(
    raw_data=raw_data,
    multiindex_data=data
)

# Results are ready to use
print(f"Selected {results['reference_sites_count']} reference sites")
print(f"Cluster distribution:\n{results['cluster_distribution']}")

# Display visualization
results['cluster_visualization_fig']
```

### Example 2: Custom Parameters

```python
results = reference_sites_taxa_assemblage_pipeline(
    raw_data=raw_data,
    multiindex_data=data,
    reference_percentile=50,  # Bottom 50% as reference
    species_transformation='chord',  # Chord transformation
    linkage_method='average',  # Average linkage
    n_clusters=4,  # 4 clusters instead of 3
    env_variables=['Measured Depth (m)', 'Temperature (oC)', 'LOI (%)'],
    create_ward_analysis=True,  # Include Ward analysis figure
    label_positions=[80, 200, 320, 450]  # Custom label positions
)
```

### Example 3: Minimal Visualization (No Diagnostics)

```python
results = reference_sites_taxa_assemblage_pipeline(
    raw_data=raw_data,
    multiindex_data=data,
    reference_percentile=52,
    species_transformation='hellinger',
    linkage_method='ward',
    n_clusters=3,
    create_cluster_visualization=True,  # Only final visualization
    create_dendrogram_comparison=False,
    create_comparison_plots=False,
    create_fusion_plots=False,
    create_ward_analysis=False,
    verbose=False  # Suppress progress messages
)
```

### Example 4: Comprehensive Diagnostics

```python
results = reference_sites_taxa_assemblage_pipeline(
    raw_data=raw_data,
    multiindex_data=data,
    reference_percentile=52,
    species_transformation='hellinger',
    linkage_method='ward',
    n_clusters=3,
    create_dendrogram_comparison=True,  # Compare 4 linkage methods
    create_comparison_plots=True,  # Detailed linkage comparisons
    create_fusion_plots=True,  # Fusion level analysis
    create_ward_analysis=True,  # Ward's method analysis
    label_positions=[90, 250, 380]
)

# Access diagnostic figures
ward_fig = results['clustering_results']['ward_analysis_fig']
comparison_fig = results['clustering_results']['comparison_plots_fig']
```

## Output Structure

The pipeline returns data with the following structure:

### `raw_data` (Single-level DataFrame)
```
Index: Site IDs
Columns:
  - Latitude, Longitude
  - Environmental variables (Depth, Velocity, DO, etc.)
  - Taxa (species abundances)
  - Pollution_Score
  - if_ref (bool): True for reference sites, False otherwise
  - clusters (float): Cluster ID (0, 1, 2, ...) for reference sites, NaN for non-reference
```

### `multiindex_data` (Multi-index DataFrame)
```
Index: Site IDs
Columns (3 levels):
  Level 0: Category ('env', 'taxa', 'chemical', 'Clusters')
  Level 1: Subcategory
  Level 2: Variable name
  
Special column:
  ('Clusters', 'Hierarchical', 'clusters'): Cluster labels
```

## Individual Components

For advanced users, individual pipeline components can be used separately:

```python
from zci.taxa_assemblage_in_refs import (
    impute_velocity_for_all_sites,
    select_reference_sites,
    compare_habitat_variables,
    cluster_species_hierarchical,
    visualize_cluster_analysis
)

# Step-by-step manual pipeline
velocity_results = impute_velocity_for_all_sites(raw_data, multiindex_data)
selection_results = select_reference_sites(raw_data, multiindex_data, percentile=52)
cluster_labels, clustering_results = cluster_species_hierarchical(taxa_data, n_clusters=3)
viz_fig = visualize_cluster_analysis(raw_data, multiindex_data)
```

## Transformation Methods

### Hellinger Transformation (Recommended)
- **Use case**: General-purpose, reduces impact of dominant species
- **Formula**: sqrt(relative abundance)
- **Best for**: Most ecological datasets with varying abundance ranges

### Chord Transformation
- **Use case**: Emphasizes presence/absence patterns
- **Formula**: Abundance / sqrt(sum of squared abundances)
- **Best for**: Datasets where rare species are important

### Octave Transformation
- **Use case**: Strong emphasis on rare species
- **Formula**: -log2(abundance + 1)
- **Best for**: Datasets focused on community diversity

## Linkage Methods

### Ward's Method (Recommended)
- Minimizes within-cluster variance
- Best for finding compact, spherical clusters
- Default choice for ecological data

### Average Linkage
- Uses average distance between all pairs
- More robust to noise than Ward
- Good for elongated clusters

### Complete Linkage
- Uses maximum distance between clusters
- Creates compact clusters
- Sensitive to outliers

### Single Linkage
- Uses minimum distance between clusters
- Can create "chaining" effect
- Rarely recommended for ecological data

## Cluster Interpretation

The output visualization includes three panels:

1. **Left Panel: Geographic Distribution**
   - Shows spatial distribution of reference sites
   - Color-coded by cluster assignment
   - Background shows lakes and rivers

2. **Upper Right: Environmental Variables**
   - Z-score standardized habitat features
   - Asymmetric error bars (± SEM)
   - Shows which habitats characterize each cluster

3. **Lower Right: Taxa Composition**
   - Hellinger-transformed abundances
   - Top N most abundant taxa
   - Shows which species characterize each cluster

## Tips and Best Practices

1. **Choosing Reference Percentile**
   - Start with 50-60% to get ~half of sites
   - Lower percentiles = stricter reference criteria
   - Check pollution score distribution first

2. **Selecting Number of Clusters**
   - Use `create_ward_analysis=True` to see fusion levels
   - Look for "elbow" in fusion level plot
   - 3-5 clusters typical for ecological data

3. **Transformation Selection**
   - Default to 'hellinger' for most cases
   - Use 'chord' if rare species are important
   - Compare transformations if uncertain

4. **Validating Results**
   - Check cluster distribution (avoid very small clusters)
   - Examine environmental gradients across clusters
   - Verify geographic coherence of clusters

## Troubleshooting

### Issue: Too few/many reference sites
**Solution**: Adjust `reference_percentile` parameter

### Issue: Unbalanced cluster sizes
**Solution**: Try different `n_clusters` or `linkage_method`

### Issue: Cluster visualization looks cluttered
**Solution**: Reduce `top_n_taxa` or specify fewer `env_variables`

### Issue: Missing environmental variables in plot
**Solution**: Check variable names with `raw_data.columns.tolist()`

## Module Files

- `pipeline.py`: Main pipeline orchestration
- `velocity_imputation.py`: Velocity value imputation
- `reference_site_selection.py`: Reference site selection and comparison
- `hierarchical_clustering.py`: Clustering algorithms and transformations
- `cluster_visualization.py`: Visualization functions
- `__init__.py`: Module exports

## Dependencies

- pandas
- numpy
- matplotlib
- scipy (hierarchical clustering)
- geopandas (map visualization)
- scikit-learn (transformations)

## Authors

Developed for St. Clair-Detroit River System Benthic Macroinvertebrate Analysis

## License

Part of the ZCI (Zoological Contamination Index) project
