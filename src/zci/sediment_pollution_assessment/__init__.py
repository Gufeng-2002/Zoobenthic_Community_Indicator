"""
Sediment pollution assessment tools for ZCI (Zoobenthic Community Indicators).

This subpackage contains modules for PCA-based pollution assessment,
chemical variable weighting, and statistical evaluation methods.
"""

from .weighted_pca import (
    PCAResult,
    weighted_PCA_computation,
    PCs_filter_loading_vs_weight,
    pollution_scores_of_subPCs,
    WeightedPCA_Scores
)

from .chemical_weights import (
    VARIABLE_TYPE_BY_NAME,
    TYPE_WEIGHTS,
    configure_type_weights,
    configure_variable_type_map,
    configure_weights_by_name,
    get_variable_weight,
    build_weights_for_columns
)

__all__ = [
    # Weighted PCA
    "PCAResult",
    "weighted_PCA_computation", 
    "PCs_filter_loading_vs_weight",
    "pollution_scores_of_subPCs",
    "WeightedPCA_Scores",
    
    # Chemical Weights
    "VARIABLE_TYPE_BY_NAME",
    "TYPE_WEIGHTS",
    "configure_type_weights",
    "configure_variable_type_map", 
    "configure_weights_by_name",
    "get_variable_weight",
    "build_weights_for_columns"
]
