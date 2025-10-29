"""
ZCI (Zhigan Chemical Index) - Sediment Pollution Assessment Framework

This package provides tools for ecological data analysis, particularly focused on
sediment contamination assessment using PCA-based methods and statistical evaluation.

Subpackages:
- data_process: Data transformation and DataFrame operations
- sediment_pollution_assessment: PCA-based pollution assessment tools  
- environmental_partition_cluster: Environmental clustering and partitioning (in development)
"""

# Import from subpackages for convenience
from .data_process import (
    hellinger_transform, log1p_standardize, log10p_standardize,
    wrap_columns, get_block, set_block, add_site_block, concat_blocks,
    flatten_columns, align_blocks_by_index, merge_into_master_by_station,
    merge_pollution_scores_into_master
)

from .sediment_pollution_assessment import (
    chemical_weights, ordination_metrices, weighted_pca
)

__all__ = [
    # Data processing
    "hellinger_transform", "log1p_standardize", "log10p_standardize",
    "wrap_columns", "get_block", "set_block", "add_site_block", "concat_blocks",
    "flatten_columns", "align_blocks_by_index", "merge_into_master_by_station",
    "merge_pollution_scores_into_master",
    
    # PCA Assessment
    "pca_chemical_assessment", "PCAContaminationResult",
    "compute_pollution_scores_with_labels", "select_pcs_by_weighted_loadings",
    
    # PCA Evaluation  
    "build_groups_from_labels", "build_groups_from_quantiles",
    "prepare_feature_matrix", "permutation_manova_euclidean",
    "evaluate_pca_assessment", "loss_from_result", 
    "directional_mean_permutation_test", "evaluate_directional_mean_test",
    "plot_permanova_null_distribution", "plot_directional_null_distribution", 
    "run_assessment_suite",
    
    # Chemical Weights
    "build_weights_for_columns", "VARIABLE_TYPE_BY_NAME", "TYPE_WEIGHTS",
    "configure_type_weights", "get_variable_weight"
]

# Package metadata
__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "ZCI (Zhigan Chemical Index) - Sediment Pollution Assessment Framework"