"""
Contamination Assessment Module

This module provides a comprehensive toolkit for assessing sediment contamination
using principal component analysis (PCA) of pollution variables.

Main Components:
----------------
1. transformations: Data transformation and preprocessing
2. pca_analysis: Principal component extraction
3. visualizations: Ridge plots and other visualizations
4. validation: RDA-based validation of pollution PCs
5. scoring: Pollution score computation
6. pipeline: High-level orchestration function

Quick Start:
------------
For most users, the pipeline function provides everything needed:

    >>> from zci.contamination_assessment import contamination_assessment_pipeline
    >>> results = contamination_assessment_pipeline(data)
    >>> pollution_scores = results['pollution_scores']

For more control, use individual functions:

    >>> from zci.contamination_assessment import (
    ...     log_z_score_transform,
    ...     pca_with_PC_loadings,
    ...     compute_pollution_scores
    ... )
    >>> transformed = log_z_score_transform(pollution_data)
    >>> PC_loadings, PC_scores = pca_with_PC_loadings(transformed)
    >>> scores = compute_pollution_scores(PC_scores)
"""

# Pipeline (main entry point)
from .pipeline import contamination_assessment_pipeline

# Transformations
from .transformations import (
    log_z_score_transform,
    get_clustered_variable_order
)

# PCA Analysis
from .pca_analysis import pca_with_PC_loadings

# Visualizations
from .visualizations import create_ridge_plot

# Validation
from .validation import rda_pollution_species_analysis

# Scoring
from .scoring import (
    compute_pollution_scores,
    merge_pollution_scores_into_data
)

# Expose all public functions
__all__ = [
    # Pipeline
    'contamination_assessment_pipeline',
    
    # Transformations
    'log_z_score_transform',
    'get_clustered_variable_order',
    
    # PCA
    'pca_with_PC_loadings',
    
    # Visualization
    'create_ridge_plot',
    
    # Validation
    'rda_pollution_species_analysis',
    
    # Scoring
    'compute_pollution_scores',
    'merge_pollution_scores_into_data',
]

# Backward compatibility: maintain old import paths
# This allows existing code using "from zci.contamination_assessment.pca_pollution_scores import ..."
# to continue working without changes
__version__ = '2.0.0'
