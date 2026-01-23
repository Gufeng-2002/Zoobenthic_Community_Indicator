"""
Taxa Assemblage in Reference Sites Module

This module provides utilities for analyzing taxa assemblages at reference sites
in the St. Clair-Detroit River System, including velocity imputation, reference
site selection, habitat comparison, and hierarchical clustering.

The main entry point is the `reference_sites_taxa_assemblage_pipeline` function,
which orchestrates the complete analysis workflow.
"""

# Pipeline - Main entry point
from .pipeline import (
    reference_sites_taxa_assemblage_pipeline,
    compare_reference_habitat_variables
)

# Individual components (for advanced users)
from .velocity_imputation import impute_velocity_for_all_sites
from .reference_site_selection import select_reference_sites, compare_habitat_variables
from .hierarchical_clustering import (
    cluster_species_hierarchical,
    hellinger_transform,
    chord_transform,
    octave_transform
)
from .cluster_visualization import (
    visualize_cluster_analysis,
    plot_cluster_dendrogram_with_map
)
from .boxcox_anova import (
    boxcox_transform_and_anova_env,
    boxcox_transform_and_anova_taxa,
    perform_boxcox_anova_analysis,
    create_anova_summary_table,
    create_anova_excel_table
)

__all__ = [
    # Pipeline functions
    'reference_sites_taxa_assemblage_pipeline',
    'compare_reference_habitat_variables',
    
    # Individual components
    'impute_velocity_for_all_sites',
    'select_reference_sites',
    'compare_habitat_variables',
    'cluster_species_hierarchical',
    'hellinger_transform',
    'chord_transform',
    'octave_transform',
    'visualize_cluster_analysis',
    'plot_cluster_dendrogram_with_map',
    
    # Box-Cox ANOVA
    'boxcox_transform_and_anova_env',
    'boxcox_transform_and_anova_taxa',
    'perform_boxcox_anova_analysis',
    'create_anova_summary_table',
    'create_anova_excel_table',
]
