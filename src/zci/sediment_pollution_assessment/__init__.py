"""
Sediment pollution assessment tools for ZCI (Zhigan Chemical Index).

This subpackage contains modules for PCA-based pollution assessment,
chemical variable weighting, and statistical evaluation methods.
"""

from .pca_assessment import *
from .pca_evaluation import *
from .chemical_weights import *

__all__ = [
    # PCA Assessment
    "pca_chemical_assessment", "PCAContaminationResult", 
    "compute_pollution_scores_with_labels", "select_pcs_by_weighted_loadings",
    
    # PCA Evaluation
    "build_groups_from_labels", "build_groups_from_quantiles",
    "prepare_feature_matrix", "permutation_manova_euclidean",
    "evaluate_pca_assessment", "loss_from_result", "EvalResult",
    "directional_mean_permutation_test", "evaluate_directional_mean_test",
    "DirMeanTestResult", "plot_permanova_null_distribution",
    "plot_directional_null_distribution", "run_assessment_suite",
    
    # Chemical Weights
    "build_weights_for_columns", "VARIABLE_TYPE_BY_NAME", "TYPE_WEIGHTS",
    "configure_type_weights", "get_variable_weight"
]
