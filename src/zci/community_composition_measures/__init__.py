"""
Community Composition Measures Module

This module provides tools for analyzing community composition using PCA-based ordination
within habitat clusters. The main workflow includes:

1. Applying separate PCAs on Hellinger-transformed species data for each cluster
2. Visualizing taxa loadings (species contributions to PCs)
3. Calculating Zoobenthic Community Indicator (ZCI) - compositional distance from reference
4. Projecting non-training sites onto fitted PC space
5. Visualizing training vs projected sites in ordination space

Main Functions:
--------------
- fit_cluster_pcas: Fit PCA models for each cluster on selected sites
- plot_taxa_loadings_stacked: Create stacked bar plots of taxa loadings
- calculate_zci: Calculate ZCI for all sites
- project_sites_to_pc_space: Project new sites using fitted PCA models
- plot_ordination_comparison: Visualize training vs projected sites
- community_composition_pipeline: Run the complete analysis pipeline

Example Usage:
-------------
>>> from zci.community_composition_measures import community_composition_pipeline
>>> results = community_composition_pipeline(
...     raw_data=raw_data,
...     multiindex_data=data,
...     cluster_column='clusters',
...     pollution_column='Pollution_Score',
...     training_percentile=52,
...     variance_threshold=0.70,
...     top_n_taxa=10
... )
"""

from .pca_analysis import (
    hellinger_transform,
    chord_transform,
    octave_transform,
    bray_curtis_transform,
    transform_taxa_data,
    fit_cluster_pcas,
    get_n_components_for_variance
)

from .taxa_loadings import (
    get_top_taxa_loadings,
    plot_taxa_loadings_stacked,
    plot_taxa_loadings_single_cluster,
    plot_taxa_loadings_consistent
)

from .zci_calculation import (
    calculate_zci_for_cluster,
    calculate_zci_all_clusters,
    project_sites_to_pc_space,
    calculate_zci_all_clusters_with_threshold
)

from .visualization import (
    plot_ordination_comparison,
    plot_zci_vs_pollution,
    create_comprehensive_figure
)

from .pc_diagnostics import (
    plot_pc_loadings_by_cluster,
    calculate_pc_pollution_regressions,
    plot_pc_pollution_regressions,
    create_pc_regression_summary_table,
    extract_significant_pc_loadings,
    calculate_pollution_pc_vs_species_pc_regressions,
    plot_pollution_pc_vs_species_pc_regressions,
    create_pollution_species_pc_summary_table
)

from .pipeline import (
    community_composition_pipeline
)

__all__ = [
    # PCA Analysis & Transformations
    'hellinger_transform',
    'chord_transform',
    'octave_transform',
    'bray_curtis_transform',
    'transform_taxa_data',
    'fit_cluster_pcas',
    'get_n_components_for_variance',
    # Taxa Loadings
    'get_top_taxa_loadings',
    'plot_taxa_loadings_stacked',
    'plot_taxa_loadings_single_cluster',
    'plot_taxa_loadings_consistent',
    # ZCI Calculation
    'calculate_zci_for_cluster',
    'calculate_zci_all_clusters',
    'project_sites_to_pc_space',
    'calculate_zci_all_clusters_with_threshold',
    # Visualization
    'plot_ordination_comparison',
    'plot_zci_vs_pollution',
    'create_comprehensive_figure',
    # PC Diagnostics
    'plot_pc_loadings_by_cluster',
    'calculate_pc_pollution_regressions',
    'plot_pc_pollution_regressions',
    'create_pc_regression_summary_table',
    'extract_significant_pc_loadings',
    'calculate_pollution_pc_vs_species_pc_regressions',
    'plot_pollution_pc_vs_species_pc_regressions',
    'create_pollution_species_pc_summary_table',
    # Pipeline
    'community_composition_pipeline'
]
