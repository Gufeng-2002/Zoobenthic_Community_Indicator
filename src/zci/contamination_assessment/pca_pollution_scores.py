# DEPRECATED MODULE - For Backward Compatibility Only
# =====================================================
# This module is deprecated as of v2.0.0 (January 2026)
#
# The contamination assessment functionality has been refactored into
# focused modules for better maintainability:
#   - transformations.py: Data transformation
#   - pca_analysis.py: PCA computation
#   - visualizations.py: Plotting functions
#   - validation.py: RDA validation
#   - scoring.py: Pollution score computation
#   - pipeline.py: Complete workflow orchestration
#
# MIGRATION:
#   Instead of: from zci.contamination_assessment.pca_pollution_scores import func
#   Use:        from zci.contamination_assessment import func
#
# This file imports and re-exports functions for backward compatibility.
# All functionality remains available but is now better organized.
# =====================================================

import warnings

# Issue deprecation warning on import
warnings.warn(
    "The 'pca_pollution_scores' module is deprecated. "
    "Please use 'from zci.contamination_assessment import ...' instead. "
    "See contamination_assessment/README.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Import all functions from new modular structure for backward compatibility
from .transformations import log_z_score_transform, get_clustered_variable_order
from .pca_analysis import pca_with_PC_loadings
from .visualizations import create_ridge_plot
from .validation import rda_pollution_species_analysis
from .scoring import compute_pollution_scores, merge_pollution_scores_into_data
from .pipeline import contamination_assessment_pipeline

# Re-export everything for backward compatibility
__all__ = [
    'log_z_score_transform',
    'get_clustered_variable_order',
    'pca_with_PC_loadings',
    'create_ridge_plot',
    'rda_pollution_species_analysis',
    'compute_pollution_scores',
    'merge_pollution_scores_into_data',
    'contamination_assessment_pipeline',
]

# Note: FIGURE_SAVE_PATH is now defined in individual modules
# but we re-export it here for compatibility
from .pca_analysis import FIGURE_SAVE_PATH
